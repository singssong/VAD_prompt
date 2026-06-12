#!/usr/bin/env python3
"""Fit positional normal-feature statistics from normal training images."""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = 256
FEATURE_SIZE = 32


def image_files(root: Path):
    return sorted(
        p
        for p in root.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.files = image_files(root)
        if not self.files:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            tensor = TF.pil_to_tensor(image).float().div_(255.0)
        tensor = TF.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return tensor


class WideResNetFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, images):
        x = self.stem(images)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer1 = F.adaptive_avg_pool2d(layer1, (FEATURE_SIZE, FEATURE_SIZE))
        layer3 = F.interpolate(
            layer3, size=(FEATURE_SIZE, FEATURE_SIZE), mode="bilinear", align_corners=False
        )
        return torch.cat((layer1, layer2, layer3), dim=1)


def smooth_maps(maps):
    kernel_size = 7
    sigma = 2.0
    coords = torch.arange(kernel_size, device=maps.device, dtype=maps.dtype)
    coords = coords - (kernel_size - 1) / 2
    kernel = torch.exp(-(coords.square()) / (2 * sigma * sigma))
    kernel = (kernel / kernel.sum()).outer(kernel / kernel.sum())
    return F.conv2d(maps, kernel.view(1, 1, kernel_size, kernel_size), padding=kernel_size // 2)


def anomaly_maps(features, mean, variance):
    distances = ((features - mean).square() / variance).mean(dim=1, keepdim=True).sqrt()
    distances = F.interpolate(
        distances, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )
    return smooth_maps(distances)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("./data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("./outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    extractor = WideResNetFeatures().to(device)

    total_channels = 256 + 512 + 1024
    generator = torch.Generator().manual_seed(0)
    channel_indices = torch.randperm(total_channels, generator=generator)[: args.channels].sort().values
    channel_indices_device = channel_indices.to(device)

    feature_sum = torch.zeros(args.channels, FEATURE_SIZE, FEATURE_SIZE, dtype=torch.float64)
    feature_square_sum = torch.zeros_like(feature_sum)
    count = 0
    with torch.inference_mode():
        for images in loader:
            features = extractor(images.to(device))[:, channel_indices_device].cpu().double()
            feature_sum += features.sum(dim=0)
            feature_square_sum += features.square().sum(dim=0)
            count += features.shape[0]

    mean = feature_sum / count
    variance = feature_square_sum / count - mean.square()
    variance_floor = variance.mean(dim=(1, 2), keepdim=True) * 0.01 + 1e-6
    variance = variance.clamp_min(variance_floor)
    mean = mean.float()
    variance = variance.float()

    # Calibrate map rendering from training data only.
    calibration_values = []
    with torch.inference_mode():
        mean_device = mean.to(device)
        variance_device = variance.to(device)
        for images in loader:
            features = extractor(images.to(device))[:, channel_indices_device]
            maps = anomaly_maps(features, mean_device, variance_device)
            calibration_values.append(maps.flatten().cpu())
    calibration_values = torch.cat(calibration_values)
    map_low = torch.quantile(calibration_values, 0.50).item()
    map_high = torch.quantile(calibration_values, 0.999).item()
    if not math.isfinite(map_high) or map_high <= map_low:
        map_high = map_low + 1.0

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "PaDiM-style positional diagonal Gaussian",
            "backbone": "wide_resnet50_2",
            "weights": "IMAGENET1K_V2",
            "image_size": IMAGE_SIZE,
            "feature_size": FEATURE_SIZE,
            "channel_indices": channel_indices,
            "mean": mean,
            "variance": variance,
            "map_low": map_low,
            "map_high": map_high,
            "train_count": count,
        },
        args.model_out,
    )
    print(f"Saved model trained on {count} images to {args.model_out}")


if __name__ == "__main__":
    main()

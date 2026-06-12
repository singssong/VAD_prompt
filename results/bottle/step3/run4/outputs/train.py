#!/usr/bin/env python3
"""Train a spatial feature-distribution anomaly detector on normal images."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = 256
FEATURE_DIM = 128
SEED = 17


class ImageDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.paths = sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            return self.transform(image), self.paths[index].name


class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        layer1 = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        size = layer1.shape[-2:]
        layer2 = F.interpolate(layer2, size=size, mode="bilinear", align_corners=False)
        layer3 = F.interpolate(layer3, size=size, mode="bilinear", align_corners=False)
        return torch.cat((layer1, layer2, layer3), dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    weights = ResNet18_Weights.DEFAULT
    dataset = ImageDataset(args.train_dir, weights.transforms(crop_size=IMAGE_SIZE, resize_size=IMAGE_SIZE))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, len(dataset)),
        pin_memory=args.device.startswith("cuda"),
    )

    backbone = resnet18(weights=weights)
    extractor = FeatureExtractor(backbone).to(args.device).eval()
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)

    generator = torch.Generator().manual_seed(SEED)
    selected = torch.randperm(64 + 128 + 256, generator=generator)[:FEATURE_DIM]
    selected_device = selected.to(args.device)

    feature_sum = None
    feature_square_sum = None
    count = 0
    with torch.inference_mode():
        for images, _ in loader:
            features = extractor(images.to(args.device, non_blocking=True))
            features = features.index_select(1, selected_device).float()
            batch_sum = features.sum(dim=0, dtype=torch.float64).cpu()
            batch_square_sum = features.square().sum(dim=0, dtype=torch.float64).cpu()
            feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
            feature_square_sum = (
                batch_square_sum if feature_square_sum is None
                else feature_square_sum + batch_square_sum
            )
            count += features.shape[0]

    mean = (feature_sum / count).float()
    variance = ((feature_square_sum - feature_sum.square() / count) / max(count - 1, 1)).float()
    # A relative floor prevents nearly constant channels from dominating the distance.
    channel_floor = variance.mean(dim=(1, 2), keepdim=True) * 0.01
    variance = variance.clamp_min(channel_floor.clamp_min(1e-6))

    calibration_samples = []
    mean_device = mean.to(args.device)
    variance_device = variance.to(args.device)
    with torch.inference_mode():
        for images, _ in loader:
            features = extractor(images.to(args.device, non_blocking=True))
            features = features.index_select(1, selected_device).float()
            distances = ((features - mean_device).square() / variance_device).mean(dim=1).sqrt()
            calibration_samples.append(distances.flatten().cpu())
    calibration = torch.quantile(torch.cat(calibration_samples), 0.999).item()

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "PaDiM-style diagonal Mahalanobis spatial feature distribution",
            "backbone": "ImageNet-pretrained ResNet-18",
            "image_size": IMAGE_SIZE,
            "selected_channels": selected,
            "mean": mean,
            "variance": variance,
            "pixel_calibration": max(calibration, 1e-6),
            "backbone_state_dict": backbone.state_dict(),
        },
        args.model_out,
    )
    print(f"Trained on {count} normal images; model saved to {args.model_out}")


if __name__ == "__main__":
    main()

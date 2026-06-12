#!/usr/bin/env python3
"""Score test images using trained spatial normal-feature statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        names = [p.name for p in self.paths]
        if len(names) != len(set(names)):
            raise RuntimeError("Test filenames must be unique")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image, [256, 256], interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.to_tensor(image)
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD), path.name


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image):
        x = self.stem(image)
        feature1 = self.layer1(x)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        size = feature1.shape[-2:]
        feature2 = F.interpolate(feature2, size=size, mode="bilinear", align_corners=False)
        feature3 = F.interpolate(feature3, size=size, mode="bilinear", align_corners=False)
        return torch.cat((feature1, feature2, feature3), dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if state["image_size"] != 256:
        raise RuntimeError("This inference script requires a 256x256 model")

    dataset = ImageDataset(args.test_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().to(device)
    channel_indices = state["channel_indices"].to(device)
    mean = state["mean"].to(device)
    variance = state["variance"].to(device)

    maps = {}
    with torch.inference_mode():
        for images, names in loader:
            features = extractor(images.to(device, non_blocking=True))
            features = features.index_select(1, channel_indices)
            distances = ((features - mean).square() / variance).mean(dim=1).sqrt()
            distances = F.interpolate(
                distances.unsqueeze(1), size=(256, 256),
                mode="bilinear", align_corners=False,
            )
            distances = F.avg_pool2d(distances, kernel_size=7, stride=1, padding=3)
            for name, anomaly_map in zip(names, distances[:, 0].cpu().numpy()):
                maps[name] = anomaly_map.astype(np.float32)

    scores = {}
    for name, anomaly_map in maps.items():
        top_count = max(1, anomaly_map.size // 100)
        top_values = np.partition(anomaly_map.ravel(), -top_count)[-top_count:]
        scores[name] = float(top_values.mean())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)

    # Use one global scale so pixel intensities remain comparable between images.
    all_pixels = np.concatenate([anomaly_map.ravel() for anomaly_map in maps.values()])
    scale_high = max(float(np.percentile(all_pixels, 99.5)), 1e-8)
    for name, anomaly_map in maps.items():
        encoded = np.clip(anomaly_map / scale_high * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(encoded, mode="L").save(pixel_dir / name)

    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as file:
        json.dump(scores, file, indent=2, sort_keys=True)
        file.write("\n")
    print(f"Scored {len(scores)} images; wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()

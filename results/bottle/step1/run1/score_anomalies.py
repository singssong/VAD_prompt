#!/usr/bin/env python3
"""Score aligned product images with a PaDiM-style spatial feature model."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path, transform) -> None:
        self.files = sorted(
            path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.files:
            raise ValueError(f"No images found in {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        with Image.open(self.files[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.files[index].name


class MultiScaleBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.layer1(self.stem(images))
        layer2 = self.layer2(features)
        layer3 = self.layer3(layer2)

        # Equalize scale contributions before joining the two receptive-field sizes.
        layer2 = F.normalize(layer2, dim=1)
        layer3 = F.normalize(layer3, dim=1)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((layer2, layer3), dim=1)


@torch.inference_mode()
def extract_features(
    loader: DataLoader,
    backbone: nn.Module,
    projection: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    batches: list[torch.Tensor] = []
    names: list[str] = []
    for images, batch_names in loader:
        images = images.to(device, non_blocking=True)
        features = backbone(images)
        features = features.permute(0, 2, 3, 1)
        features = features @ projection
        batches.append(features.cpu())
        names.extend(batch_names)
    return torch.cat(batches), names


def robust_spatial_model(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate location-wise center and scale while limiting outlier influence."""
    center = features.median(dim=0).values
    absolute_deviation = (features - center).abs()
    scale = 1.4826 * absolute_deviation.median(dim=0).values

    # A global channel floor prevents near-constant dimensions from dominating.
    scale_floor = scale.flatten(0, 2).median(dim=0).values.clamp_min(1e-3)
    scale = torch.maximum(scale, 0.25 * scale_floor.view(1, 1, -1))
    return center, scale.clamp_min(1e-4)


def score_features(
    features: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    squared_z = ((features - center) / scale).square()
    maps = squared_z.mean(dim=-1).sqrt()

    # Small spatial tolerance suppresses harmless one-cell alignment differences.
    maps = -F.max_pool2d(-maps.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    tail_count = max(1, math.ceil(maps[0].numel() * 0.02))
    scores = maps.flatten(1).topk(tail_count, dim=1).values.mean(dim=1)
    return scores, maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    train_data = ImageDataset(args.train_dir, weights.transforms())
    test_data = ImageDataset(args.test_dir, weights.transforms())
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": min(4, len(train_data)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, **loader_options)
    test_loader = DataLoader(test_data, **loader_options)

    backbone = MultiScaleBackbone().eval().to(device)
    source_dim = 512 + 1024
    generator = torch.Generator(device=device).manual_seed(args.seed)
    projection = torch.randn(
        source_dim, args.feature_dim, generator=generator, device=device
    ) / math.sqrt(args.feature_dim)

    train_features, _ = extract_features(train_loader, backbone, projection, device)
    center, scale = robust_spatial_model(train_features)
    del train_features

    test_features, names = extract_features(test_loader, backbone, projection, device)
    scores, _ = score_features(test_features, center, scale)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "anomaly_score"))
        writer.writerows((name, f"{score:.10f}") for name, score in zip(names, scores.tolist()))

    print(f"Wrote {len(names)} scores to {args.output}")
    print("Method: robust spatial PaDiM-style multi-scale feature modeling")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

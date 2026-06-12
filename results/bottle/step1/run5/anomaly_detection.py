#!/usr/bin/env python3
"""Train-only PaDiM-style anomaly scoring for the bottle images."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path, transform: object) -> None:
        self.paths = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"No images found in {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


class MultiScaleFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        first = self.layer1(x)
        second = self.layer2(first)
        third = self.layer3(second)
        target_size = second.shape[-2:]
        first = F.adaptive_avg_pool2d(first, target_size)
        third = F.interpolate(third, size=target_size, mode="bilinear", align_corners=False)
        return torch.cat((first, second, third), dim=1)


def collect_descriptors(
    loader: DataLoader,
    model: nn.Module,
    selected_channels: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], tuple[int, int]]:
    descriptors: list[torch.Tensor] = []
    names: list[str] = []
    spatial_size: tuple[int, int] | None = None

    with torch.inference_mode():
        for images, batch_names in loader:
            features = model(images.to(device, non_blocking=True))
            features = features.index_select(1, selected_channels)
            spatial_size = (features.shape[-2], features.shape[-1])
            descriptors.append(features.permute(0, 2, 3, 1).flatten(1, 2).cpu())
            names.extend(batch_names)

    if spatial_size is None:
        raise RuntimeError("No descriptors were extracted")
    return torch.cat(descriptors), names, spatial_size


def fit_distribution(train_features: torch.Tensor, regularization: float) -> tuple[torch.Tensor, torch.Tensor]:
    # Shape is [images, spatial locations, descriptor dimensions].
    train_features = train_features.float()
    mean = train_features.mean(dim=0)
    centered = train_features - mean
    covariance = torch.einsum("nld,nle->lde", centered, centered)
    covariance /= max(train_features.shape[0] - 1, 1)
    identity = torch.eye(covariance.shape[-1], dtype=covariance.dtype)
    covariance += regularization * identity.unsqueeze(0)
    precision = torch.linalg.inv(covariance)
    return mean, precision


def score_images(
    features: torch.Tensor,
    mean: torch.Tensor,
    precision: torch.Tensor,
    top_fraction: float,
) -> np.ndarray:
    centered = features.float() - mean.unsqueeze(0)
    squared_distance = torch.einsum("bld,lde,ble->bl", centered, precision, centered)
    distances = squared_distance.clamp_min_(0).sqrt_()
    top_count = max(1, round(distances.shape[1] * top_fraction))
    # Averaging the strongest local responses is more stable than a single-pixel maximum.
    return distances.topk(top_count, dim=1).values.mean(dim=1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dimensions", type=int, default=100)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--top-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    train_dataset = ImageDataset(args.train_dir, weights.transforms())
    test_dataset = ImageDataset(args.test_dir, weights.transforms())
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    model = MultiScaleFeatures().eval().to(device)
    total_channels = 256 + 512 + 1024
    if not 1 <= args.dimensions <= total_channels:
        raise ValueError(f"--dimensions must be between 1 and {total_channels}")
    selected = torch.randperm(total_channels, device=device)[: args.dimensions]

    train_features, _, spatial_size = collect_descriptors(train_loader, model, selected, device)
    test_features, test_names, _ = collect_descriptors(test_loader, model, selected, device)
    mean, precision = fit_distribution(train_features, args.regularization)
    scores = score_images(test_features, mean, precision, args.top_fraction)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("filename", "anomaly_score"))
        writer.writerows((name, f"{float(score):.10f}") for name, score in zip(test_names, scores))

    print(f"Scored {len(test_names)} images at feature-map size {spatial_size}.")
    print(f"Output: {args.output}")
    print("Method: PaDiM-style spatial Gaussian modeling with top-region Mahalanobis scoring")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score bottle images using a spatial Gaussian model of pretrained patch features."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class FeatureExtractor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        size = layer1.shape[-2:]
        return torch.cat(
            [
                layer1,
                F.interpolate(layer2, size=size, mode="bilinear", align_corners=False),
                F.interpolate(layer3, size=size, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )


def image_paths(directory: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in extensions
    )


@torch.inference_mode()
def extract_features(
    paths: list[Path],
    model: torch.nn.Module,
    transform,
    selected_channels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(
        ImageDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    output = []
    for images in loader:
        features = model(images.to(device, non_blocking=True))
        output.append(features[:, selected_channels].cpu())
    return torch.cat(output)


def fit_spatial_gaussian(
    features: torch.Tensor, regularization: float
) -> tuple[torch.Tensor, torch.Tensor]:
    # [N, C, H, W] -> one C-dimensional normal distribution per H,W location.
    samples = features.permute(2, 3, 0, 1).reshape(-1, features.shape[0], features.shape[1])
    mean = samples.mean(dim=1)
    centered = samples - mean[:, None, :]
    covariance = centered.transpose(1, 2) @ centered / (samples.shape[1] - 1)
    diagonal_scale = covariance.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    eye = torch.eye(features.shape[1], dtype=features.dtype)[None]
    covariance += regularization * diagonal_scale[:, None, None] * eye
    precision_cholesky = torch.linalg.cholesky(covariance)
    return mean, precision_cholesky


def score_features(
    features: torch.Tensor,
    mean: torch.Tensor,
    precision_cholesky: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    samples = features.permute(0, 2, 3, 1).reshape(features.shape[0], -1, features.shape[1])
    scores = []
    for sample in samples:
        delta = sample - mean
        whitened = torch.linalg.solve_triangular(
            precision_cholesky, delta.unsqueeze(-1), upper=False
        ).squeeze(-1)
        patch_scores = whitened.square().sum(dim=-1).sqrt()
        # Gaussian smoothing suppresses isolated feature noise while retaining defects.
        side = int(patch_scores.numel() ** 0.5)
        patch_scores = patch_scores.reshape(1, 1, side, side)
        patch_scores = F.avg_pool2d(patch_scores, kernel_size=3, stride=1, padding=1)
        scores.append(torch.quantile(patch_scores.flatten(), quantile))
    return torch.stack(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=100)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--quantile", type=float, default=0.995)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths = image_paths(args.train_dir)
    test_paths = image_paths(args.test_dir)
    if not train_paths or not test_paths:
        raise RuntimeError("Both train and test image directories must contain images")

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    model = FeatureExtractor().eval().to(device)
    total_channels = 256 + 512 + 1024
    generator = torch.Generator().manual_seed(0)
    selected = torch.randperm(total_channels, generator=generator)[: args.channels]

    print(f"Extracting features from {len(train_paths)} training images on {device}...")
    train_features = extract_features(
        train_paths, model, weights.transforms(), selected.to(device), device, args.batch_size
    )
    print("Fitting spatial Gaussian distributions...")
    mean, precision_cholesky = fit_spatial_gaussian(
        train_features, args.regularization
    )
    del train_features

    print(f"Scoring {len(test_paths)} test images...")
    test_features = extract_features(
        test_paths, model, weights.transforms(), selected.to(device), device, args.batch_size
    )
    scores = score_features(
        test_features, mean, precision_cholesky, args.quantile
    ).numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["filename", "anomaly_score"])
        writer.writerows(
            (path.name, f"{score:.8f}") for path, score in zip(test_paths, scores)
        )
    print(f"Wrote {len(scores)} scores to {args.output}")
    print("Method: spatial Gaussian patch anomaly detection (PaDiM)")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

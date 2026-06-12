#!/usr/bin/env python3
"""Unsupervised image anomaly scoring using spatial Gaussian feature models."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


class FeatureExtractor(nn.Module):
    """Wide ResNet stem and first three residual stages."""

    def __init__(self) -> None:
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        model = wide_resnet50_2(weights=weights)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return layer1, layer2, layer3


@torch.inference_mode()
def fit_statistics(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    sums: list[torch.Tensor] | None = None
    squared_sums: list[torch.Tensor] | None = None
    count = 0

    for images, _ in loader:
        features = model(images.to(device))
        if sums is None:
            sums = [torch.zeros_like(feature[0], dtype=torch.float64) for feature in features]
            squared_sums = [torch.zeros_like(value) for value in sums]
        assert squared_sums is not None
        for index, feature in enumerate(features):
            feature64 = feature.double()
            sums[index] += feature64.sum(dim=0)
            squared_sums[index] += feature64.square().sum(dim=0)
        count += images.shape[0]

    if not count or sums is None or squared_sums is None:
        raise ValueError("No training images were found")

    statistics = []
    for total, squared_total in zip(sums, squared_sums):
        mean = total / count
        variance = (squared_total / count - mean.square()).clamp_min(1e-6)
        statistics.append((mean.float(), variance.float()))
    return statistics


def top_tail_mean(anomaly_map: torch.Tensor, fraction: float = 0.02) -> torch.Tensor:
    flattened = anomaly_map.flatten(start_dim=1)
    k = max(1, round(flattened.shape[1] * fraction))
    return flattened.topk(k, dim=1).values.mean(dim=1)


@torch.inference_mode()
def raw_scores(
    model: nn.Module,
    loader: DataLoader,
    statistics: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    batches: list[np.ndarray] = []

    for images, batch_names in loader:
        features = model(images.to(device))
        layer_scores = []
        for feature, (mean, variance) in zip(features, statistics):
            standardized_error = (feature - mean).square() / variance
            anomaly_map = standardized_error.mean(dim=1)
            layer_scores.append(top_tail_mean(anomaly_map))
        batches.append(torch.stack(layer_scores, dim=1).cpu().numpy())
        names.extend(batch_names)

    return names, np.concatenate(batches, axis=0)


def robust_calibrate(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    median = np.median(train_scores, axis=0)
    mad = np.median(np.abs(train_scores - median), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-6)
    return (scores - median) / scale


def write_scores(path: Path, names: list[str], scores: np.ndarray) -> None:
    order = np.argsort(scores)[::-1]
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)

    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["image", "anomaly_score", "anomaly_rank"])
        for name, score, rank in zip(names, scores, ranks):
            writer.writerow([name, f"{float(score):.10f}", int(rank)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    train_paths = image_files(args.train_dir)
    test_paths = image_files(args.test_dir)
    if not train_paths:
        raise ValueError(f"No images found in {args.train_dir}")
    if not test_paths:
        raise ValueError(f"No images found in {args.test_dir}")

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    transform = weights.transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 2,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(ImageDataset(train_paths, transform), **loader_options)
    test_loader = DataLoader(ImageDataset(test_paths, transform), **loader_options)

    model = FeatureExtractor().eval().to(device)
    statistics = fit_statistics(model, train_loader, device)
    _, train_raw = raw_scores(model, train_loader, statistics, device)
    test_names, test_raw = raw_scores(model, test_loader, statistics, device)

    calibrated = robust_calibrate(train_raw, test_raw)
    final_scores = calibrated.mean(axis=1)
    write_scores(args.output, test_names, final_scores)

    print(f"Scored {len(test_names)} images -> {args.output}")
    print("Method: multi-scale spatial diagonal Gaussian feature modeling")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

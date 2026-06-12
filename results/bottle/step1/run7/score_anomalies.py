#!/usr/bin/env python3
"""Score test images with a location-aware pretrained feature model."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, directory: Path, transform) -> None:
        self.paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise ValueError(f"No images found in {directory}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


class PatchFeatureExtractor(nn.Module):
    """Wide ResNet feature extractor with fused 32x32 feature maps."""

    def __init__(self) -> None:
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2

    def forward(self, images: Tensor) -> Tensor:
        features1 = self.layer1(self.stem(images))
        features2 = self.layer2(features1)
        features1 = F.avg_pool2d(features1, kernel_size=3, stride=2, padding=1)
        return torch.cat((features1, features2), dim=1)


@torch.inference_mode()
def fit_statistics(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[Tensor, Tensor]:
    feature_sum: Tensor | None = None
    feature_sq_sum: Tensor | None = None
    count = 0

    for images, _ in loader:
        features = model(images.to(device, non_blocking=True)).float()
        batch_sum = features.sum(dim=0)
        batch_sq_sum = features.square().sum(dim=0)
        feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
        feature_sq_sum = (
            batch_sq_sum if feature_sq_sum is None else feature_sq_sum + batch_sq_sum
        )
        count += features.shape[0]

    if feature_sum is None or feature_sq_sum is None or count < 2:
        raise ValueError("At least two training images are required")

    mean = feature_sum / count
    variance = (feature_sq_sum / count - mean.square()).clamp_min(1e-5)

    # Prevent nearly constant channels from dominating standardized distances.
    channel_floor = variance.flatten(1).median(dim=1).values[:, None, None] * 0.05
    variance = torch.maximum(variance, channel_floor.clamp_min(1e-5))
    return mean, variance


def reduce_anomaly_map(anomaly_map: Tensor) -> Tensor:
    """Average the strongest local responses for an image-level score."""
    anomaly_map = F.avg_pool2d(anomaly_map[:, None], 3, stride=1, padding=1)[:, 0]
    flat = anomaly_map.flatten(1)
    top_count = max(1, round(flat.shape[1] * 0.02))
    return flat.topk(top_count, dim=1).values.mean(dim=1)


@torch.inference_mode()
def score_loader(
    model: nn.Module,
    loader: DataLoader,
    mean: Tensor,
    variance: Tensor,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    scores: list[float] = []
    for images, batch_names in loader:
        features = model(images.to(device, non_blocking=True)).float()
        anomaly_map = ((features - mean).square() / variance).mean(dim=1)
        batch_scores = reduce_anomaly_map(anomaly_map)
        names.extend(batch_names)
        scores.extend(batch_scores.cpu().tolist())
    return names, np.asarray(scores, dtype=np.float64)


def robust_normalize(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    center = float(np.median(reference))
    mad = float(np.median(np.abs(reference - center)))
    scale = max(1.4826 * mad, float(reference.std()) * 0.1, 1e-8)
    z_scores = (scores - center) / scale
    # A smooth positive scale is easier to consume while preserving score order.
    return np.logaddexp(0.0, z_scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    train_data = ImageDataset(args.train_dir, weights.transforms())
    test_data = ImageDataset(args.test_dir, weights.transforms())
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": min(4, args.batch_size),
        "pin_memory": torch.cuda.is_available(),
        "shuffle": False,
    }
    train_loader = DataLoader(train_data, **loader_args)
    test_loader = DataLoader(test_data, **loader_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchFeatureExtractor().eval().to(device)
    mean, variance = fit_statistics(model, train_loader, device)
    _, train_raw_scores = score_loader(model, train_loader, mean, variance, device)
    test_names, test_raw_scores = score_loader(
        model, test_loader, mean, variance, device
    )
    anomaly_scores = robust_normalize(test_raw_scores, train_raw_scores)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "anomaly_score"])
        writer.writerows(
            (name, f"{score:.10f}")
            for name, score in zip(test_names, anomaly_scores, strict=True)
        )

    metadata = {
        "method": "location-aware diagonal Gaussian patch feature modeling",
        "backbone": "ImageNet-pretrained Wide ResNet-50-2",
        "train_images": len(train_data),
        "test_images": len(test_data),
        "score_file": str(args.output),
        "higher_score_is_more_anomalous": True,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Unsupervised image anomaly scoring with a PatchCore-style memory bank."""

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


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, directory: Path, transform: nn.Module) -> None:
        self.paths = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise ValueError(f"No images found in {directory}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


class PatchFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        model = wide_resnet50_2(weights=weights)
        self.stem = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = self.layer1(features)
        layer2 = self.layer2(features)
        layer3 = self.layer3(layer2)

        # Local averaging makes patch descriptors less sensitive to carpet fibers.
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        patches = torch.cat((layer2, layer3), dim=1)
        patches = patches.permute(0, 2, 3, 1).flatten(1, 2)
        return F.normalize(patches, dim=-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def build_memory_bank(
    extractor: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bank_size: int,
    seed: int,
) -> torch.Tensor:
    batches = []
    for images, _ in loader:
        batches.append(extractor(images.to(device)).cpu())
    patches = torch.cat(batches, dim=0).flatten(0, 1)
    generator = torch.Generator().manual_seed(seed)
    count = min(bank_size, len(patches))
    indices = torch.randperm(len(patches), generator=generator)[:count]
    return patches[indices].contiguous().to(device)


def nearest_distances(
    queries: torch.Tensor, memory_bank: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    results = []
    # For unit vectors, squared Euclidean distance is 2 - 2 * cosine similarity.
    memory_t = memory_bank.T.contiguous()
    for start in range(0, len(queries), chunk_size):
        query = queries[start : start + chunk_size]
        max_similarity = torch.matmul(query, memory_t).amax(dim=1)
        results.append(torch.sqrt((2.0 - 2.0 * max_similarity).clamp_min(0)))
    return torch.cat(results)


@torch.inference_mode()
def score_images(
    extractor: nn.Module,
    loader: DataLoader,
    memory_bank: torch.Tensor,
    device: torch.device,
    chunk_size: int,
    top_fraction: float,
) -> list[tuple[str, float]]:
    scores: list[tuple[str, float]] = []
    for images, names in loader:
        patches = extractor(images.to(device))
        for image_patches, name in zip(patches, names):
            distances = nearest_distances(image_patches, memory_bank, chunk_size)
            top_count = max(1, round(len(distances) * top_fraction))
            score = distances.topk(top_count).values.mean().item()
            scores.append((name, score))
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bank-size", type=int, default=30000)
    parser.add_argument("--distance-chunk-size", type=int, default=1024)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if not 0 < args.top_fraction <= 1:
        raise ValueError("--top-fraction must be in (0, 1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    transform = weights.transforms(crop_size=256, resize_size=256)
    train_data = ImageDataset(args.train_dir, transform)
    test_data = ImageDataset(args.test_dir, transform)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda"
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda"
    )

    extractor = PatchFeatureExtractor().eval().to(device)
    memory_bank = build_memory_bank(
        extractor, train_loader, device, args.bank_size, args.seed
    )
    scores = score_images(
        extractor, test_loader, memory_bank, device,
        args.distance_chunk_size, args.top_fraction
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filename", "anomaly_score"))
        writer.writerows((name, f"{score:.8f}") for name, score in scores)

    print(f"Scored {len(scores)} images -> {args.output}")
    print("Method: PatchCore-style nearest-neighbor patch memory bank")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

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


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
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
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features2 = self.layer2(self.stem(images))
        features3 = self.layer3(features2)
        features3 = F.interpolate(
            features3, size=features2.shape[-2:], mode="bilinear", align_corners=False
        )

        # Normalize each semantic level so neither dominates by feature magnitude.
        features2 = F.normalize(features2, dim=1)
        features3 = F.normalize(features3, dim=1)
        features = torch.cat((features2, features3), dim=1)
        features = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        features = F.normalize(features, dim=1)
        return features.permute(0, 2, 3, 1).flatten(1, 2)


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


@torch.inference_mode()
def build_memory_bank(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    patches_per_image: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    sampled_batches: list[torch.Tensor] = []
    for images, _ in loader:
        patches = model(images.to(device, non_blocking=True))
        sample_count = min(patches_per_image, patches.shape[1])
        selected = []
        for image_patches in patches:
            indices = torch.randperm(
                image_patches.shape[0], generator=generator, device=device
            )[:sample_count]
            selected.append(image_patches[indices])
        sampled_batches.append(torch.cat(selected).cpu())
    return torch.cat(sampled_batches).contiguous()


def nearest_cosine_distances(
    query: torch.Tensor, memory: torch.Tensor, memory_chunk_size: int = 4096
) -> torch.Tensor:
    best_similarity = torch.full(
        (query.shape[0],), -1.0, dtype=query.dtype, device=query.device
    )
    for start in range(0, memory.shape[0], memory_chunk_size):
        chunk = memory[start : start + memory_chunk_size].to(query.device)
        best_similarity = torch.maximum(
            best_similarity, (query @ chunk.T).amax(dim=1)
        )
    return 1.0 - best_similarity


@torch.inference_mode()
def score_images(
    model: nn.Module,
    loader: DataLoader,
    memory: torch.Tensor,
    device: torch.device,
    top_fraction: float,
) -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []
    for images, names in loader:
        batch_patches = model(images.to(device, non_blocking=True))
        for patches, name in zip(batch_patches, names):
            distances = nearest_cosine_distances(patches, memory)
            top_count = max(1, round(distances.numel() * top_fraction))
            score = distances.topk(top_count).values.mean().item()
            results.append((name, score))
    return results


def robust_calibrate(
    scores: list[tuple[str, float]], reference_values: np.ndarray
) -> list[tuple[str, float, float]]:
    median = float(np.median(reference_values))
    mad = float(np.median(np.abs(reference_values - median)))
    scale = max(1.4826 * mad, 1e-8)
    return [
        (name, max(0.0, (raw_score - median) / scale), raw_score)
        for name, raw_score in scores
    ]


def make_loader(
    paths: list[Path], transform, batch_size: int, workers: int
) -> DataLoader:
    return DataLoader(
        ImageDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patches-per-image", type=int, default=160)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths = image_paths(args.train_dir)
    test_paths = image_paths(args.test_dir)
    if not train_paths or not test_paths:
        raise RuntimeError("Both train and test image directories must contain images")

    shuffled_train = train_paths.copy()
    random.Random(args.seed).shuffle(shuffled_train)
    calibration_count = max(1, round(len(shuffled_train) * args.calibration_fraction))
    calibration_paths = shuffled_train[:calibration_count]
    memory_paths = shuffled_train[calibration_count:]

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    transform = weights.transforms(crop_size=256, resize_size=256)
    model = PatchFeatureExtractor().to(device)

    memory = build_memory_bank(
        model,
        make_loader(memory_paths, transform, args.batch_size, args.workers),
        device,
        args.patches_per_image,
        args.seed,
    )
    calibration_scores = score_images(
        model,
        make_loader(calibration_paths, transform, args.batch_size, args.workers),
        memory,
        device,
        args.top_fraction,
    )
    test_scores = score_images(
        model,
        make_loader(test_paths, transform, args.batch_size, args.workers),
        memory,
        device,
        args.top_fraction,
    )

    calibrated = robust_calibrate(
        test_scores, np.array([score for _, score in calibration_scores])
    )
    with args.output.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("filename", "anomaly_score", "raw_score"))
        writer.writerows(
            (name, f"{score:.10f}", f"{raw_score:.10f}")
            for name, score, raw_score in calibrated
        )

    print(f"Scored {len(test_scores)} images -> {args.output}")
    print("Method: PatchCore-style patch memory bank with robust normal-score calibration")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2 (layer2 + layer3)")


if __name__ == "__main__":
    main()

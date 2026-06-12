#!/usr/bin/env python3
"""Score test images with a training-only PatchCore-style memory bank."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_paths(directory: Path) -> list[Path]:
    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No images found in {directory}")
    return paths


class PatchFeatureExtractor(torch.nn.Module):
    def __init__(self, projection_dim: int = 384, seed: int = 13) -> None:
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        model = wide_resnet50_2(weights=weights)
        self.stem = torch.nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool, model.layer1
        )
        self.layer2 = model.layer2
        self.layer3 = model.layer3

        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(1536, projection_dim, generator=generator)
        projection /= math.sqrt(projection_dim)
        self.register_buffer("projection", projection)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = (images - self.mean) / self.std
        x = self.stem(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)

        # Local averaging gives each patch context while retaining small defects.
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        features = torch.cat((layer2, layer3), dim=1)
        features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        features = features @ self.projection
        return F.normalize(features, dim=1)


def load_batch(paths: list[Path], device: torch.device) -> torch.Tensor:
    images = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        images.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(images).to(device, non_blocking=True)


@torch.inference_mode()
def build_memory_bank(
    extractor: PatchFeatureExtractor,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    patches_per_image: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    sampled_features = []

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        features = extractor(load_batch(batch_paths, device))
        patches_per_feature_map = features.shape[0] // len(batch_paths)
        features = features.view(len(batch_paths), patches_per_feature_map, -1)

        count = min(patches_per_image, patches_per_feature_map)
        for image_features in features:
            indices = torch.randperm(
                patches_per_feature_map, generator=generator, device=device
            )[:count]
            sampled_features.append(image_features[indices].cpu())

        print(f"Memory features: {min(start + batch_size, len(paths))}/{len(paths)}")

    return torch.cat(sampled_features).contiguous()


def nearest_distances(
    query: torch.Tensor,
    memory_bank: torch.Tensor,
    memory_chunk_size: int = 4096,
) -> torch.Tensor:
    nearest = torch.full(
        (query.shape[0],), float("inf"), dtype=query.dtype, device=query.device
    )
    for start in range(0, memory_bank.shape[0], memory_chunk_size):
        memory_chunk = memory_bank[start : start + memory_chunk_size].to(
            query.device, non_blocking=True
        )
        # Features are unit vectors, so this is Euclidean distance without cdist.
        similarities = query @ memory_chunk.T
        chunk_nearest = torch.sqrt(
            (2.0 - 2.0 * similarities.max(dim=1).values).clamp_min(0.0)
        )
        nearest = torch.minimum(nearest, chunk_nearest)
    return nearest


@torch.inference_mode()
def score_images(
    extractor: PatchFeatureExtractor,
    paths: list[Path],
    memory_bank: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    scores = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        features = extractor(load_batch(batch_paths, device))
        patches_per_image = features.shape[0] // len(batch_paths)
        features = features.view(len(batch_paths), patches_per_image, -1)

        for path, image_features in zip(batch_paths, features):
            distances = nearest_distances(image_features, memory_bank)
            top_k = max(1, math.ceil(0.01 * distances.numel()))
            score = distances.topk(top_k).values.mean().item()
            scores.append(score)
            print(f"Scored {path.name}: {score:.6f}")
    return scores


def robust_unit_scores(raw_scores: list[float]) -> list[float]:
    values = np.asarray(raw_scores, dtype=np.float64)
    low, high = np.percentile(values, [5.0, 95.0])
    if high <= low:
        return [0.0] * len(raw_scores)
    return np.clip((values - low) / (high - low), 0.0, 1.0).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patches-per-image", type=int, default=80)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    train_paths = image_paths(args.train_dir)
    test_paths = image_paths(args.test_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}; {len(train_paths)} train and {len(test_paths)} test images")

    extractor = PatchFeatureExtractor(seed=args.seed).to(device)
    memory_bank = build_memory_bank(
        extractor,
        train_paths,
        device,
        args.batch_size,
        args.patches_per_image,
        args.seed,
    )
    print(f"Memory bank shape: {tuple(memory_bank.shape)}")

    raw_scores = score_images(
        extractor, test_paths, memory_bank, device, args.batch_size
    )
    unit_scores = robust_unit_scores(raw_scores)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "anomaly_score", "raw_score"])
        for path, unit_score, raw_score in zip(test_paths, unit_scores, raw_scores):
            writer.writerow([path.name, f"{unit_score:.10f}", f"{raw_score:.10f}"])

    print(f"Wrote {len(raw_scores)} scores to {args.output}")
    print("Method: PatchCore-style patch memory bank with exact nearest neighbors")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

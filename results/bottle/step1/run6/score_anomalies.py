#!/usr/bin/env python3
"""Score test images with a training-only PatchCore-style memory bank."""

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
    def __init__(self, root: Path, transform: object) -> None:
        self.paths = sorted(
            path
            for path in root.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and "checkpoint" not in path.name.lower()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


class FeatureExtractor(nn.Module):
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
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer2 = F.adaptive_avg_pool2d(layer2, layer3.shape[-2:])
        return torch.cat((layer2, layer3), dim=1)


def make_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    return projection / np.sqrt(output_dim)


@torch.inference_mode()
def collect_training_patches(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    projection: torch.Tensor,
) -> torch.Tensor:
    batches = []
    projection = projection.to(device)
    for images, _ in loader:
        features = model(images.to(device, non_blocking=True))
        patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        patches = F.normalize(patches @ projection, dim=1)
        batches.append(patches.cpu())
    return torch.cat(batches)


def sample_memory_bank(
    patches: torch.Tensor, memory_size: int, seed: int
) -> torch.Tensor:
    if len(patches) <= memory_size:
        return patches
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(patches), generator=generator)[:memory_size]
    return patches[indices]


def nearest_distances(
    queries: torch.Tensor, memory: torch.Tensor, chunk_size: int = 4096
) -> torch.Tensor:
    distances = []
    for chunk in queries.split(chunk_size):
        # Unit-normalized features: squared L2 distance = 2 - 2 * cosine similarity.
        best_similarity = chunk @ memory.T
        distances.append(torch.sqrt((2 - 2 * best_similarity.max(dim=1).values).clamp_min(0)))
    return torch.cat(distances)


@torch.inference_mode()
def score_images(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    projection: torch.Tensor,
    memory: torch.Tensor,
    top_fraction: float,
) -> list[tuple[str, float]]:
    projection = projection.to(device)
    memory = memory.to(device)
    results = []
    for images, names in loader:
        features = model(images.to(device, non_blocking=True))
        batch_size, channels, height, width = features.shape
        patches = features.permute(0, 2, 3, 1).reshape(-1, channels)
        patches = F.normalize(patches @ projection, dim=1)
        patch_distances = nearest_distances(patches, memory).reshape(batch_size, -1)
        top_k = max(1, int(round(patch_distances.shape[1] * top_fraction)))
        scores = patch_distances.topk(top_k, dim=1).values.mean(dim=1)
        results.extend((name, float(score)) for name, score in zip(names, scores.cpu()))
    return results


def write_scores(output: Path, scores: list[tuple[str, float]]) -> None:
    raw = np.asarray([score for _, score in scores], dtype=np.float64)
    low, high = np.percentile(raw, [5, 95])
    normalized = np.clip((raw - low) / max(high - low, 1e-12), 0.0, 1.0)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "anomaly_score", "raw_score"])
        for (name, score), norm_score in zip(scores, normalized):
            writer.writerow([name, f"{norm_score:.10f}", f"{score:.10f}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--memory-size", type=int, default=30000)
    parser.add_argument("--projection-dim", type=int, default=384)
    parser.add_argument("--top-fraction", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    train_data = ImageDataset(args.train_dir, weights.transforms())
    test_data = ImageDataset(args.test_dir, weights.transforms())
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    model = FeatureExtractor().eval().to(device)
    projection = make_projection(1536, args.projection_dim, args.seed)
    patches = collect_training_patches(model, train_loader, device, projection)
    memory = sample_memory_bank(patches, args.memory_size, args.seed)
    scores = score_images(
        model, test_loader, device, projection, memory, args.top_fraction
    )
    write_scores(args.output, scores)
    print(f"Scored {len(scores)} images; wrote {args.output}")
    print(f"Training patches: {len(patches)}; memory bank: {len(memory)}")
    print("Method: PatchCore-style patch nearest-neighbor anomaly detection")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

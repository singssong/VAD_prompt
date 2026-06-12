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
from torchvision import transforms
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path, transform: object) -> None:
        self.paths = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
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


class PatchEmbeddingExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = wide_resnet50_2(
            weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2
        )
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.layer1(self.stem(images))
        layer2 = self.layer2(features)
        layer3 = self.layer3(layer2)
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat((layer2, layer3), dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--memory-size", type=int, default=30000)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def extract_embeddings(
    loader: DataLoader,
    model: nn.Module,
    projection: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], int]:
    all_embeddings: list[torch.Tensor] = []
    all_names: list[str] = []
    patches_per_image = 0

    for images, names in loader:
        features = model(images.to(device, non_blocking=True))
        features = features.permute(0, 2, 3, 1).flatten(1, 2)
        embeddings = features @ projection
        embeddings = F.normalize(embeddings, dim=-1)
        patches_per_image = embeddings.shape[1]
        all_embeddings.append(embeddings.cpu())
        all_names.extend(names)

    return torch.cat(all_embeddings), all_names, patches_per_image


def nearest_memory_distance(
    queries: torch.Tensor,
    memory: torch.Tensor,
    device: torch.device,
    query_chunk_size: int = 4096,
    memory_chunk_size: int = 8192,
) -> torch.Tensor:
    distances: list[torch.Tensor] = []
    memory = memory.to(device)

    with torch.inference_mode():
        for start in range(0, len(queries), query_chunk_size):
            query = queries[start:start + query_chunk_size].to(device)
            best_similarity = torch.full(
                (len(query),), -1.0, device=device, dtype=torch.float32
            )
            for memory_start in range(0, len(memory), memory_chunk_size):
                memory_chunk = memory[memory_start:memory_start + memory_chunk_size]
                similarities = query @ memory_chunk.T
                best_similarity = torch.maximum(
                    best_similarity, similarities.max(dim=1).values
                )
            distances.append((1.0 - best_similarity).cpu())

    return torch.cat(distances)


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    train_dataset = ImageDataset(args.train_dir, transform)
    test_dataset = ImageDataset(args.test_dir, transform)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 2,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model = PatchEmbeddingExtractor().eval().to(device)
    feature_dim = 512 + 1024
    generator = torch.Generator(device=device).manual_seed(args.seed)
    projection = torch.randn(
        feature_dim, args.projection_dim, generator=generator, device=device
    ) / np.sqrt(args.projection_dim)

    print(f"Extracting patches from {len(train_dataset)} training images...")
    train_embeddings, _, _ = extract_embeddings(
        train_loader, model, projection, device
    )
    train_patches = train_embeddings.flatten(0, 1)
    memory_size = min(args.memory_size, len(train_patches))
    sample_generator = torch.Generator().manual_seed(args.seed)
    memory_indices = torch.randperm(
        len(train_patches), generator=sample_generator
    )[:memory_size]
    memory = train_patches[memory_indices].contiguous()

    print(
        f"Scoring {len(test_dataset)} test images against "
        f"{len(memory)} normal patches..."
    )
    test_embeddings, names, patches_per_image = extract_embeddings(
        test_loader, model, projection, device
    )
    patch_distances = nearest_memory_distance(
        test_embeddings.flatten(0, 1), memory, device
    ).reshape(len(test_dataset), patches_per_image)

    top_k = max(1, round(patches_per_image * args.top_fraction))
    raw_scores = patch_distances.topk(top_k, dim=1).values.mean(dim=1)
    order = torch.argsort(raw_scores)
    percentile_scores = torch.empty_like(raw_scores)
    if len(raw_scores) == 1:
        percentile_scores[0] = 0.0
    else:
        percentile_scores[order] = torch.linspace(0.0, 1.0, len(raw_scores))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("filename", "anomaly_score", "raw_score"))
        for name, percentile, raw in zip(names, percentile_scores, raw_scores):
            writer.writerow((name, f"{percentile.item():.10f}", f"{raw.item():.10f}"))

    print(f"Wrote {len(names)} scores to {args.output}")
    print("Method: PatchCore-style nearest-neighbor patch memory bank")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

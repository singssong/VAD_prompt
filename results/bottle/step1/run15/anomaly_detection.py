#!/usr/bin/env python3
"""Reference-only image anomaly scoring with a PatchCore-style memory bank."""

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


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path, transform) -> None:
        self.paths = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


class PatchFeatures(nn.Module):
    """Wide ResNet feature extractor for local appearance patches."""

    def __init__(self) -> None:
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = self.layer1(features)
        layer2 = self.layer2(features)
        layer3 = self.layer3(layer2)

        # Local averaging makes descriptors less sensitive to pixel-level noise.
        layer2 = F.avg_pool2d(layer2, 3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, 3, stride=1, padding=1)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        patches = torch.cat((layer2, layer3), dim=1)
        patches = patches.permute(0, 2, 3, 1).flatten(1, 2)
        return F.normalize(patches, dim=-1)


@torch.inference_mode()
def build_memory_bank(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    patches_per_image: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    samples = []
    for images, _ in loader:
        patches = model(images.to(device, non_blocking=True))
        count = min(patches_per_image, patches.shape[1])
        for image_patches in patches:
            indices = torch.randperm(
                image_patches.shape[0], generator=generator, device=device
            )[:count]
            samples.append(image_patches[indices])
    return torch.cat(samples).contiguous()


def nearest_distances(
    queries: torch.Tensor, memory: torch.Tensor, chunk_size: int = 4096
) -> torch.Tensor:
    """Cosine distance to the nearest normalized reference descriptor."""
    output = []
    memory_t = memory.T.contiguous()
    for start in range(0, queries.shape[0], chunk_size):
        similarities = queries[start : start + chunk_size] @ memory_t
        output.append(1.0 - similarities.max(dim=1).values)
    return torch.cat(output)


@torch.inference_mode()
def score_images(
    model: nn.Module,
    loader: DataLoader,
    memory: torch.Tensor,
    device: torch.device,
    top_fraction: float,
) -> list[tuple[str, float]]:
    results = []
    for images, names in loader:
        patches = model(images.to(device, non_blocking=True))
        for image_patches, name in zip(patches, names):
            distances = nearest_distances(image_patches, memory)
            top_count = max(1, round(distances.numel() * top_fraction))
            score = distances.topk(top_count).values.mean().item()
            results.append((name, score))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("scores.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patches-per-image", type=int, default=32)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    transform = weights.transforms(crop_size=256, resize_size=256)
    train_data = ImageDataset(args.train_dir, transform)
    test_data = ImageDataset(args.test_dir, transform)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": min(4, args.batch_size),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=False, **loader_args)
    test_loader = DataLoader(test_data, shuffle=False, **loader_args)

    model = PatchFeatures().eval().to(device)
    memory = build_memory_bank(
        model, train_loader, device, args.patches_per_image, args.seed
    )
    scores = score_images(
        model, test_loader, memory, device, args.top_fraction
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filename", "anomaly_score"))
        writer.writerows((name, f"{score:.10f}") for name, score in scores)

    print(f"Scored {len(scores)} images -> {args.output}")
    print(f"Reference memory: {memory.shape[0]} patches x {memory.shape[1]} features")
    print("Method: PatchCore-style patch nearest-neighbor anomaly detection")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

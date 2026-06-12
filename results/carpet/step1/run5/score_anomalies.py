#!/usr/bin/env python3
"""Score test images with a compact PatchCore-style anomaly detector."""

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
from torchvision.transforms import Compose, Normalize, ToTensor


SEED = 20260611
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.transform = Compose(
            [
                ToTensor(),
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


class WideResNetFeatures(nn.Module):
    """Wide ResNet trunk returning aligned layer1 and layer2 patch features."""

    def __init__(self) -> None:
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)

        # Local average pooling adds neighborhood context while retaining defects.
        layer1 = F.avg_pool2d(layer1, kernel_size=3, stride=1, padding=1)
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer1 = F.interpolate(
            layer1, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat((layer1, layer2), dim=1)


def image_paths(directory: Path) -> list[Path]:
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"No images found in {directory}")
    return paths


def make_projection(
    input_dim: int, output_dim: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(SEED)
    matrix = torch.randn(
        input_dim, output_dim, generator=generator, device=device, dtype=torch.float32
    )
    matrix, _ = torch.linalg.qr(matrix, mode="reduced")
    return matrix.to(torch.float16 if device.type == "cuda" else torch.float32)


@torch.inference_mode()
def extract_features(
    paths: list[Path],
    model: nn.Module,
    projection: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, list[str], tuple[int, int]]:
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    all_features: list[torch.Tensor] = []
    names: list[str] = []
    grid_size: tuple[int, int] | None = None

    for images, batch_names in loader:
        images = images.to(device, non_blocking=True)
        features = model(images)
        batch, channels, height, width = features.shape
        grid_size = (height, width)
        features = features.permute(0, 2, 3, 1).reshape(-1, channels)
        features = features.to(projection.dtype) @ projection
        all_features.append(features.float().cpu())
        names.extend(batch_names)

    assert grid_size is not None
    stacked = torch.cat(all_features).reshape(len(paths), -1, projection.shape[1])
    return stacked, names, grid_size


def make_memory_bank(features: torch.Tensor, bank_size: int) -> torch.Tensor:
    flat = features.reshape(-1, features.shape[-1])
    generator = torch.Generator().manual_seed(SEED)
    count = min(bank_size, len(flat))
    indices = torch.randperm(len(flat), generator=generator)[:count]
    return flat[indices].contiguous()


@torch.inference_mode()
def nearest_neighbor_distances(
    queries: torch.Tensor,
    bank: torch.Tensor,
    device: torch.device,
    query_chunk: int = 512,
    bank_chunk: int = 8192,
) -> torch.Tensor:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    bank = bank.to(device=device, dtype=dtype)
    bank_norms = bank.float().square().sum(dim=1)
    output: list[torch.Tensor] = []

    for start in range(0, len(queries), query_chunk):
        query = queries[start : start + query_chunk].to(device=device, dtype=dtype)
        query_norms = query.float().square().sum(dim=1)
        best_squared_distance = torch.full(
            (len(query),), float("inf"), device=device, dtype=torch.float32
        )
        for bank_start in range(0, len(bank), bank_chunk):
            reference = bank[bank_start : bank_start + bank_chunk]
            dot_products = (query @ reference.T).float()
            squared_distances = (
                query_norms[:, None]
                + bank_norms[bank_start : bank_start + len(reference)][None, :]
                - 2.0 * dot_products
            )
            best_squared_distance = torch.minimum(
                best_squared_distance, squared_distances.amin(dim=1)
            )
        output.append(best_squared_distance.clamp_min(0).sqrt().cpu())
    return torch.cat(output)


def score_images(
    test_features: torch.Tensor,
    bank: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    image_count, patches_per_image, feature_dim = test_features.shape
    distances = nearest_neighbor_distances(
        test_features.reshape(-1, feature_dim), bank, device
    ).reshape(image_count, patches_per_image)

    # Average the strongest 1% of local responses. This is less noise-sensitive
    # than a maximum while preserving sensitivity to small defects.
    top_k = max(1, round(patches_per_image * 0.01))
    strongest = distances.topk(top_k, dim=1).values
    scores = 0.8 * strongest.mean(dim=1) + 0.2 * strongest[:, 0]
    return scores.numpy()


def write_scores(output: Path, names: list[str], scores: np.ndarray) -> None:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filename", "anomaly_score"))
        for name, score in zip(names, scores, strict=True):
            writer.writerow((name, f"{float(score):.10f}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--bank-size", type=int, default=30000)
    parser.add_argument("--projection-dim", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths = image_paths(args.train_dir)
    test_paths = image_paths(args.test_dir)
    model = WideResNetFeatures().eval().to(device)

    sample = torch.zeros(1, 3, 256, 256, device=device)
    with torch.inference_mode():
        input_dim = model(sample).shape[1]
    projection = make_projection(input_dim, args.projection_dim, device)

    print(f"Extracting reference features from {len(train_paths)} images...")
    train_features, _, grid_size = extract_features(
        train_paths, model, projection, device, args.batch_size
    )
    bank = make_memory_bank(train_features, args.bank_size)
    del train_features

    print(f"Extracting test features from {len(test_paths)} images...")
    test_features, names, _ = extract_features(
        test_paths, model, projection, device, args.batch_size
    )
    print(
        f"Scoring {len(test_paths)} images against {len(bank)} reference patches "
        f"on a {grid_size[0]}x{grid_size[1]} feature grid..."
    )
    scores = score_images(test_features, bank, device)
    write_scores(args.output, names, scores)

    print(f"Wrote {len(scores)} scores to {args.output}")
    print("Method: PatchCore-style local nearest-neighbor anomaly detection")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

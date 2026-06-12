#!/usr/bin/env python3
"""Score test images with a lightweight PatchCore-style anomaly detector."""

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
    def __init__(self, root: Path, transform) -> None:
        self.paths = sorted(
            path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
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


class MultiScaleFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        model = wide_resnet50_2(weights=weights)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = self.layer1(features)
        layer2 = self.layer2(features)
        layer3 = self.layer3(layer2)

        # Local averaging makes carpet texture descriptors less sensitive to
        # single-pixel phase shifts while retaining compact defects.
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat((layer2, layer3), dim=1)
        return features.permute(0, 2, 3, 1).flatten(1, 2)


def make_transform():
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]

    def transform(image: Image.Image) -> torch.Tensor:
        image = image.resize((256, 256), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - mean) / std

    return transform


@torch.inference_mode()
def collect_features(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, list[str], int]:
    batches = []
    names: list[str] = []
    patches_per_image = 0
    for images, batch_names in loader:
        features = model(images.to(device, non_blocking=True))
        patches_per_image = features.shape[1]
        batches.append(features.cpu())
        names.extend(batch_names)
    return torch.cat(batches), names, patches_per_image


def build_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    projection /= np.sqrt(output_dim)
    return projection


def project_and_normalize(features: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    features = features @ projection
    return F.normalize(features, dim=-1)


def build_memory_bank(
    train_features: torch.Tensor,
    projection: torch.Tensor,
    bank_size: int,
    seed: int,
) -> torch.Tensor:
    flat = train_features.flatten(0, 1)
    generator = torch.Generator().manual_seed(seed)
    if len(flat) > bank_size:
        indices = torch.randperm(len(flat), generator=generator)[:bank_size]
        flat = flat[indices]
    return project_and_normalize(flat, projection)


@torch.inference_mode()
def score_images(
    test_features: torch.Tensor,
    memory_bank: torch.Tensor,
    projection: torch.Tensor,
    device: torch.device,
    top_fraction: float,
    query_chunk: int,
) -> np.ndarray:
    memory_bank = memory_bank.to(device)
    scores = []
    for image_features in test_features:
        queries = project_and_normalize(image_features, projection).to(device)
        nearest_distances = []
        for start in range(0, len(queries), query_chunk):
            similarities = queries[start : start + query_chunk] @ memory_bank.T
            nearest_distances.append(1.0 - similarities.max(dim=1).values)
        distances = torch.cat(nearest_distances)
        top_k = max(1, round(len(distances) * top_fraction))
        # Combining a robust upper-tail mean with the maximum rewards both
        # extended defects and small, sharply localized defects.
        tail_mean = distances.topk(top_k).values.mean()
        score = 0.8 * tail_mean + 0.2 * distances.max()
        scores.append(float(score.cpu()))
    return np.asarray(scores, dtype=np.float64)


def write_scores(path: Path, names: list[str], scores: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filename", "anomaly_score"))
        for name, score in zip(names, scores, strict=True):
            writer.writerow((name, f"{score:.10f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--bank-size", type=int, default=50000)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = make_transform()
    train_data = ImageDataset(args.train_dir, transform)
    test_data = ImageDataset(args.test_dir, transform)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": min(4, args.batch_size),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    model = MultiScaleFeatures().eval().to(device)
    train_features, _, patches_per_image = collect_features(model, train_loader, device)
    test_features, test_names, _ = collect_features(model, test_loader, device)

    projection = build_projection(train_features.shape[-1], args.projection_dim, args.seed)
    memory_bank = build_memory_bank(train_features, projection, args.bank_size, args.seed)
    del train_features, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    scores = score_images(
        test_features,
        memory_bank,
        projection,
        device,
        args.top_fraction,
        query_chunk=128,
    )
    write_scores(args.output, test_names, scores)
    print(f"Scored {len(test_names)} images on {device}.")
    print(f"Training memory bank: {len(memory_bank)} patches; {patches_per_image} patches/image.")
    print(f"Wrote {args.output}")
    print("Method: PatchCore-style multi-scale patch nearest-neighbor anomaly scoring")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

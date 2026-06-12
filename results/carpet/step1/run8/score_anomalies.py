#!/usr/bin/env python3
"""Score test images with a PatchCore-style unsupervised anomaly detector."""

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
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, directory: Path, transform: nn.Module) -> None:
        self.paths = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
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


class FeatureExtractor(nn.Module):
    """Extract and align layer2/layer3 local features."""

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
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        # Local average pooling suppresses texture phase noise while retaining defects.
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        return torch.cat((layer2, layer3), dim=1)


def make_projection(
    input_dim: int, output_dim: int, device: torch.device, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    projection = torch.randn(
        input_dim, output_dim, generator=generator, device=device
    )
    return projection / math.sqrt(output_dim)


@torch.inference_mode()
def extract_embeddings(
    loader: DataLoader,
    extractor: nn.Module,
    projection: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    all_embeddings: list[torch.Tensor] = []
    all_names: list[str] = []
    for images, names in loader:
        features = extractor(images.to(device, non_blocking=True))
        patches = features.permute(0, 2, 3, 1).flatten(1, 2)
        patches = F.normalize(patches, dim=-1)
        embeddings = patches @ projection
        embeddings = F.normalize(embeddings, dim=-1)
        all_embeddings.append(embeddings.cpu())
        all_names.extend(names)
    return torch.cat(all_embeddings), all_names


def build_memory_bank(
    train_embeddings: torch.Tensor, bank_size: int, seed: int
) -> torch.Tensor:
    patches = train_embeddings.flatten(0, 1)
    if len(patches) <= bank_size:
        return patches
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(patches), generator=generator)[:bank_size]
    return patches[indices]


@torch.inference_mode()
def score_embeddings(
    embeddings: torch.Tensor,
    memory_bank: torch.Tensor,
    device: torch.device,
    image_batch_size: int = 4,
    bank_chunk_size: int = 3000,
) -> np.ndarray:
    scores: list[torch.Tensor] = []
    memory_bank = memory_bank.to(device)
    for start in range(0, len(embeddings), image_batch_size):
        batch = embeddings[start:start + image_batch_size].to(device)
        flat = batch.flatten(0, 1)
        nearest = torch.full((len(flat),), float("inf"), device=device)
        for bank_start in range(0, len(memory_bank), bank_chunk_size):
            bank = memory_bank[bank_start:bank_start + bank_chunk_size]
            # Unit vectors: squared Euclidean distance is 2 - 2*cosine similarity.
            similarities = flat @ bank.T
            nearest = torch.minimum(nearest, 2.0 - 2.0 * similarities.max(dim=1).values)
        patch_scores = nearest.clamp_min(0).sqrt().view(len(batch), -1)
        top_k = max(1, math.ceil(patch_scores.shape[1] * 0.01))
        image_scores = patch_scores.topk(top_k, dim=1).values.mean(dim=1)
        scores.append(image_scores.cpu())
    return torch.cat(scores).numpy()


def robust_normalize(scores: np.ndarray) -> np.ndarray:
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    scale = max(1.4826 * mad, 1e-8)
    z_scores = (scores - median) / scale
    return 1.0 / (1.0 + np.exp(-np.clip(z_scores, -30.0, 30.0)))


def write_scores(
    output_path: Path, names: list[str], raw_scores: np.ndarray
) -> None:
    normalized = robust_normalize(raw_scores)
    with output_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("filename", "anomaly_score", "raw_score"))
        for name, score, raw_score in zip(names, normalized, raw_scores):
            writer.writerow((name, f"{score:.10f}", f"{raw_score:.10f}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bank-size", type=int, default=12000)
    parser.add_argument("--projection-dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    transform = weights.transforms(crop_size=256, resize_size=256)
    train_loader = DataLoader(
        ImageDataset(args.train_dir, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ImageDataset(args.test_dir, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    extractor = FeatureExtractor().eval().to(device)
    projection = make_projection(1536, args.projection_dim, device, args.seed)
    train_embeddings, _ = extract_embeddings(
        train_loader, extractor, projection, device
    )
    test_embeddings, test_names = extract_embeddings(
        test_loader, extractor, projection, device
    )
    memory_bank = build_memory_bank(train_embeddings, args.bank_size, args.seed)
    raw_scores = score_embeddings(test_embeddings, memory_bank, device)
    write_scores(args.output, test_names, raw_scores)

    print(f"Wrote {len(test_names)} scores to {args.output}")
    print("Method: PatchCore-style nearest-neighbor patch memory bank")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

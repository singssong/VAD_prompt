#!/usr/bin/env python3
"""Score test images with an unsupervised PatchCore-style anomaly detector."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import Compose, Normalize, ToTensor


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.paths = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
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


def build_extractor(device: torch.device) -> torch.nn.Module:
    model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
    extractor = create_feature_extractor(
        model,
        return_nodes={"layer2": "layer2", "layer3": "layer3"},
    )
    return extractor.eval().to(device)


def make_patch_features(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    layer2 = outputs["layer2"]
    layer3 = outputs["layer3"]
    layer3 = F.interpolate(
        layer3,
        size=layer2.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    # Normalize each scale independently so neither layer dominates by magnitude.
    layer2 = F.normalize(layer2, dim=1)
    layer3 = F.normalize(layer3, dim=1)
    features = torch.cat((layer2, layer3), dim=1)
    features = features.permute(0, 2, 3, 1).flatten(1, 2)
    return F.normalize(features, dim=-1)


@torch.inference_mode()
def build_memory_bank(
    extractor: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    patches_per_image: int,
    generator: torch.Generator,
) -> torch.Tensor:
    sampled_batches: list[torch.Tensor] = []
    for images, _ in loader:
        features = make_patch_features(extractor(images.to(device)))
        patch_count = features.shape[1]
        keep = min(patches_per_image, patch_count)
        for image_features in features:
            indices = torch.randperm(patch_count, generator=generator)[:keep]
            sampled_batches.append(image_features[indices.to(device)].cpu())

    memory_bank = torch.cat(sampled_batches, dim=0)
    return memory_bank.to(device=device, dtype=torch.float16)


def nearest_neighbor_distances(
    queries: torch.Tensor,
    memory_bank: torch.Tensor,
    query_chunk_size: int = 256,
) -> torch.Tensor:
    distances: list[torch.Tensor] = []
    queries = queries.to(dtype=torch.float16)
    for chunk in queries.split(query_chunk_size):
        max_similarity = torch.full(
            (chunk.shape[0],),
            -1.0,
            device=chunk.device,
            dtype=torch.float32,
        )
        # Splitting the bank bounds temporary matrix memory on smaller GPUs.
        for bank_chunk in memory_bank.split(8192):
            similarities = chunk @ bank_chunk.T
            max_similarity = torch.maximum(
                max_similarity,
                similarities.max(dim=1).values.float(),
            )
        distances.append(1.0 - max_similarity)
    return torch.cat(distances)


@torch.inference_mode()
def score_images(
    extractor: torch.nn.Module,
    loader: DataLoader,
    memory_bank: torch.Tensor,
    device: torch.device,
    top_fraction: float,
) -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []
    for images, names in loader:
        features = make_patch_features(extractor(images.to(device)))
        for image_features, name in zip(features, names):
            patch_scores = nearest_neighbor_distances(image_features, memory_bank)
            top_k = max(1, round(patch_scores.numel() * top_fraction))
            # A small upper-tail mean is robust to isolated activation noise while
            # retaining sensitivity to localized defects.
            score = patch_scores.topk(top_k).values.mean().item()
            results.append((name, score))
    return sorted(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patches-per-image", type=int, default=128)
    parser.add_argument("--top-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260611)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(args.seed)

    train_dataset = ImageDataset(args.train_dir)
    test_dataset = ImageDataset(args.test_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    extractor = build_extractor(device)
    memory_bank = build_memory_bank(
        extractor,
        train_loader,
        device,
        args.patches_per_image,
        generator,
    )
    results = score_images(
        extractor,
        test_loader,
        memory_bank,
        device,
        args.top_fraction,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("filename", "anomaly_score"))
        writer.writerows((name, f"{score:.10f}") for name, score in results)

    print(f"Scored {len(results)} images -> {args.output}")
    print("Method: PatchCore-style patch nearest-neighbor anomaly detection")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

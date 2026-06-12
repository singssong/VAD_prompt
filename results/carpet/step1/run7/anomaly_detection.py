#!/usr/bin/env python3
"""Training-only PatchCore-style anomaly scoring for the image dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import v2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, directory: Path, transform: nn.Module) -> None:
        self.paths = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {directory}")
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
        medium = self.layer2(features)
        coarse = self.layer3(medium)
        medium = F.avg_pool2d(medium, kernel_size=3, stride=1, padding=1)
        coarse = F.avg_pool2d(coarse, kernel_size=3, stride=1, padding=1)
        coarse = F.interpolate(
            coarse, size=medium.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat((medium, coarse), dim=1)


@torch.inference_mode()
def extract_embeddings(
    loader: DataLoader,
    extractor: nn.Module,
    projection: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    batches: list[torch.Tensor] = []
    names: list[str] = []
    for images, batch_names in loader:
        features = extractor(images.to(device, non_blocking=True))
        features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        embeddings = features @ projection
        batches.append(embeddings.cpu())
        names.extend(batch_names)
    return torch.cat(batches), names


@torch.inference_mode()
def score_embeddings(
    embeddings: torch.Tensor,
    bank: torch.Tensor,
    patches_per_image: int,
    device: torch.device,
    query_chunk: int = 2048,
) -> torch.Tensor:
    nearest_parts: list[torch.Tensor] = []
    bank = bank.to(device)
    for start in range(0, len(embeddings), query_chunk):
        query = embeddings[start : start + query_chunk].to(device)
        nearest_parts.append(torch.cdist(query, bank).amin(dim=1).cpu())

    patch_scores = torch.cat(nearest_parts).reshape(-1, patches_per_image)
    tail_size = max(1, int(round(patches_per_image * 0.01)))
    return patch_scores.topk(tail_size, dim=1).values.mean(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--bank-size", type=int, default=15_000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = v2.Compose(
        [
            v2.Resize((256, 256), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
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
    source_dim = 512 + 1024
    projection = torch.randn(
        source_dim, args.embedding_dim, device=device
    ) / np.sqrt(args.embedding_dim)

    train_embeddings, _ = extract_embeddings(
        train_loader, extractor, projection, device
    )
    mean = train_embeddings.mean(dim=0)
    std = train_embeddings.std(dim=0).clamp_min(1e-6)
    train_embeddings = (train_embeddings - mean) / std

    generator = torch.Generator().manual_seed(args.seed)
    bank_count = min(args.bank_size, len(train_embeddings))
    bank_indices = torch.randperm(len(train_embeddings), generator=generator)[:bank_count]
    memory_bank = train_embeddings[bank_indices].contiguous()
    del train_embeddings

    test_embeddings, test_names = extract_embeddings(
        test_loader, extractor, projection, device
    )
    test_embeddings = (test_embeddings - mean) / std
    patches_per_image = len(test_embeddings) // len(test_names)
    scores = score_embeddings(
        test_embeddings, memory_bank, patches_per_image, device
    ).numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "anomaly_score"])
        writer.writerows(
            (name, f"{float(score):.10f}")
            for name, score in zip(test_names, scores, strict=True)
        )

    json_output = args.output.with_suffix(".json")
    with json_output.open("w") as handle:
        json.dump(
            {name: float(score) for name, score in zip(test_names, scores, strict=True)},
            handle,
            indent=2,
            sort_keys=True,
        )

    print(f"Scored {len(test_names)} images on {device}.")
    print(f"CSV: {args.output}")
    print(f"JSON: {json_output}")
    print("Method: PatchCore-style nearest-neighbor patch anomaly detection")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

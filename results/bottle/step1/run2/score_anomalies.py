#!/usr/bin/env python3
"""Score test images with a train-only PatchCore-style anomaly detector."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import v2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(directory: Path) -> list[Path]:
    """Return only direct, visible image files from a requested data directory."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


class ImageDataset(Dataset):
    def __init__(self, files: list[Path], transform: v2.Compose) -> None:
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.files[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


def make_extractor(device: torch.device) -> torch.nn.Module:
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    model = wide_resnet50_2(weights=weights)
    extractor = create_feature_extractor(
        model, return_nodes={"layer2": "layer2", "layer3": "layer3"}
    )
    return extractor.eval().to(device)


@torch.inference_mode()
def embeddings(
    extractor: torch.nn.Module,
    images: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    features = extractor(images)
    layer2 = features["layer2"]
    layer3 = F.interpolate(
        features["layer3"],
        size=layer2.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    combined = torch.cat((layer2, layer3), dim=1)
    combined = F.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)
    patches = combined.permute(0, 2, 3, 1) @ projection
    return F.normalize(patches, dim=-1)


def nearest_distances(
    query: torch.Tensor,
    memory: torch.Tensor,
    query_chunk: int = 2048,
    memory_chunk: int = 8192,
) -> torch.Tensor:
    """Cosine distance to the nearest memory patch without a huge distance matrix."""
    results = []
    for start in range(0, len(query), query_chunk):
        q = query[start : start + query_chunk]
        best_similarity = torch.full(
            (len(q),), -1.0, device=query.device, dtype=query.dtype
        )
        for memory_start in range(0, len(memory), memory_chunk):
            bank = memory[memory_start : memory_start + memory_chunk]
            best_similarity = torch.maximum(
                best_similarity, (q @ bank.T).amax(dim=1)
            )
        results.append(1.0 - best_similarity)
    return torch.cat(results)


def score_images(
    loader: DataLoader,
    extractor: torch.nn.Module,
    projection: torch.Tensor,
    memory: torch.Tensor,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    scores: list[float] = []
    for batch, batch_names in loader:
        batch = batch.to(device, non_blocking=True)
        batch_embeddings = embeddings(extractor, batch, projection)
        for image_embeddings in batch_embeddings:
            distances = nearest_distances(image_embeddings.reshape(-1, 256), memory)
            # A top-tail mean is stable while retaining sensitivity to small defects.
            top_k = max(1, int(0.01 * distances.numel()))
            score = distances.topk(top_k).values.mean()
            scores.append(float(score.cpu()))
        names.extend(batch_names)
    return names, np.asarray(scores, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--memory-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_files = image_files(args.train_dir)
    test_files = image_files(args.test_dir)
    if not train_files or not test_files:
        raise RuntimeError("Both train and test directories must contain images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((256, 256), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    train_loader = DataLoader(
        ImageDataset(train_files, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ImageDataset(test_files, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    extractor = make_extractor(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    projection = torch.randn(
        1536, 256, generator=generator, device=device
    ) / np.sqrt(256)

    memory_parts = []
    with torch.inference_mode():
        for batch, _ in train_loader:
            batch = batch.to(device, non_blocking=True)
            memory_parts.append(embeddings(extractor, batch, projection).flatten(0, 2))
    full_memory = torch.cat(memory_parts)
    sample_count = min(args.memory_size, len(full_memory))
    selected = torch.randperm(
        len(full_memory), generator=generator, device=device
    )[:sample_count]
    memory = full_memory[selected].contiguous()

    names, raw_scores = score_images(
        test_loader, extractor, projection, memory, device
    )
    # A monotonic robust calibration makes scores easier to interpret.
    train_names, train_scores = score_images(
        train_loader, extractor, projection, memory, device
    )
    del train_names
    center = np.median(train_scores)
    scale = np.median(np.abs(train_scores - center)) * 1.4826
    scale = max(scale, 1e-8)
    calibrated = (raw_scores - center) / scale

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "anomaly_score"])
        writer.writerows(
            (name, f"{score:.10f}") for name, score in zip(names, calibrated)
        )

    print(f"Wrote {len(names)} scores to {args.output}")
    print("Method: PatchCore-style nearest-neighbor patch anomaly detection")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

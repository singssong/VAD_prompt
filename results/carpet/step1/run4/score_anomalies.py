#!/usr/bin/env python3
"""Score test images with an unsupervised PatchCore-style memory bank."""

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
    def __init__(self, root: Path, transform) -> None:
        self.paths = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.paths[index].name


class FeatureExtractor(nn.Module):
    """Wide ResNet trunk exposing the two PatchCore feature levels."""

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
        level2 = self.layer2(features)
        level3 = self.layer3(level2)

        # Local averaging makes descriptors less sensitive to one-pixel shifts.
        level2 = F.avg_pool2d(level2, kernel_size=3, stride=1, padding=1)
        level3 = F.avg_pool2d(level3, kernel_size=3, stride=1, padding=1)
        level3 = F.interpolate(
            level3, size=level2.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat((level2, level3), dim=1)


def make_projection(
    input_dim: int, output_dim: int, device: torch.device, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    projection = torch.randn(
        input_dim, output_dim, generator=generator, device=device
    )
    return projection / np.sqrt(output_dim)


@torch.inference_mode()
def extract_patches(
    loader: DataLoader,
    model: nn.Module,
    projection: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], int]:
    batches: list[torch.Tensor] = []
    names: list[str] = []
    patches_per_image = 0

    for images, batch_names in loader:
        images = images.to(device, non_blocking=True)
        features = model(images)
        patches_per_image = features.shape[-2] * features.shape[-1]
        patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        patches = patches @ projection
        patches = F.normalize(patches, dim=1)
        batches.append(patches.cpu())
        names.extend(batch_names)

    return torch.cat(batches), names, patches_per_image


def build_memory(
    patches: torch.Tensor, memory_size: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    count = min(memory_size, len(patches))
    indices = torch.randperm(len(patches), generator=generator)[:count]
    return patches[indices].contiguous()


@torch.inference_mode()
def nearest_distances(
    queries: torch.Tensor,
    memory: torch.Tensor,
    device: torch.device,
    query_batch_size: int = 1024,
) -> torch.Tensor:
    memory_gpu = memory.to(device)
    distances: list[torch.Tensor] = []
    for start in range(0, len(queries), query_batch_size):
        query = queries[start:start + query_batch_size].to(device)
        # Unit-normalized descriptors: cosine distance is 1 - dot product.
        similarity = query @ memory_gpu.T
        distances.append((1.0 - similarity.max(dim=1).values).cpu())
    return torch.cat(distances)


def image_scores(
    patch_distances: torch.Tensor, image_count: int, patches_per_image: int
) -> torch.Tensor:
    distances = patch_distances.reshape(image_count, patches_per_image)
    top_k = max(1, int(round(patches_per_image * 0.01)))
    return distances.topk(top_k, dim=1).values.mean(dim=1)


def robust_normalize(scores: torch.Tensor, reference_scores: torch.Tensor) -> torch.Tensor:
    median = reference_scores.median()
    mad = (reference_scores - median).abs().median().clamp_min(1e-8)
    robust_z = (scores - median) / (1.4826 * mad)
    # A monotonic 0..1 scale, calibrated only from normal training images.
    return torch.sigmoid(robust_z)


def write_scores(
    output: Path, names: list[str], raw: torch.Tensor, calibrated: torch.Tensor
) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "anomaly_score", "calibrated_score"))
        for name, raw_score, calibrated_score in zip(
            names, raw.tolist(), calibrated.tolist()
        ):
            writer.writerow(
                (name, f"{raw_score:.10f}", f"{calibrated_score:.10f}")
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--memory-size", type=int, default=15000)
    parser.add_argument("--projection-dim", type=int, default=192)
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
        "shuffle": False,
        "num_workers": 4,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, **loader_options)
    test_loader = DataLoader(test_data, **loader_options)

    model = FeatureExtractor().eval().to(device)
    projection = make_projection(1536, args.projection_dim, device, args.seed)

    train_patches, train_names, patches_per_image = extract_patches(
        train_loader, model, projection, device
    )
    memory = build_memory(train_patches, args.memory_size, args.seed)
    train_distances = nearest_distances(train_patches, memory, device)
    train_scores = image_scores(
        train_distances, len(train_names), patches_per_image
    )

    test_patches, test_names, test_patches_per_image = extract_patches(
        test_loader, model, projection, device
    )
    if test_patches_per_image != patches_per_image:
        raise RuntimeError("Train and test feature grids have different dimensions")
    test_distances = nearest_distances(test_patches, memory, device)
    test_scores = image_scores(test_distances, len(test_names), patches_per_image)
    normalized_scores = robust_normalize(test_scores, train_scores)

    write_scores(args.output, test_names, test_scores, normalized_scores)
    print(f"Scored {len(test_names)} images on {device}.")
    print(f"Wrote {args.output}")
    print("Method: PatchCore-style patch memory bank with cosine nearest neighbors")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

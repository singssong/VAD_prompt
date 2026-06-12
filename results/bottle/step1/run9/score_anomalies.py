#!/usr/bin/env python3
"""Score test images against normal training images without using labels."""

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
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


class ImageDataset(Dataset[tuple[Tensor, str]]):
    def __init__(self, paths: list[Path], transform: object) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
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

    def forward(self, images: Tensor) -> Tensor:
        features = self.layer1(self.stem(images))
        layer2 = self.layer2(features)
        layer3 = self.layer3(layer2)

        # Local averaging gives each descriptor a stable neighborhood context.
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat((layer2, layer3), dim=1)


@torch.inference_mode()
def extract_features(
    paths: list[Path],
    model: nn.Module,
    transform: object,
    device: torch.device,
    batch_size: int,
) -> tuple[Tensor, list[str]]:
    loader = DataLoader(
        ImageDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    batches: list[Tensor] = []
    names: list[str] = []
    for images, batch_names in loader:
        feature_map = model(images.to(device, non_blocking=True))
        batches.append(feature_map.cpu())
        names.extend(batch_names)
    return torch.cat(batches), names


def project_and_normalize(
    train: Tensor, test: Tensor, output_dim: int, seed: int
) -> tuple[Tensor, Tensor]:
    channels = train.shape[1]
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(channels, output_dim, generator=generator)
    projection /= math.sqrt(output_dim)

    def apply(features: Tensor) -> Tensor:
        descriptors = features.permute(0, 2, 3, 1).contiguous()
        descriptors = descriptors @ projection
        return F.normalize(descriptors, p=2, dim=-1)

    return apply(train), apply(test)


@torch.inference_mode()
def nearest_distances(
    queries: Tensor,
    references: Tensor,
    device: torch.device,
    exclude_diagonal: bool = False,
    position_chunk: int = 64,
) -> Tensor:
    """Find nearest reference descriptor at each aligned spatial position."""
    query_count, height, width, _ = queries.shape
    reference_count = references.shape[0]
    queries = queries.reshape(query_count, height * width, -1)
    references = references.reshape(reference_count, height * width, -1)
    output = torch.empty(query_count, height * width)

    for start in range(0, height * width, position_chunk):
        stop = min(start + position_chunk, height * width)
        query = queries[:, start:stop].to(device)
        reference = references[:, start:stop].to(device)
        # Unit-normalized descriptor squared distance is 2 - 2*cosine.
        similarity = torch.einsum("qpd,rpd->qpr", query, reference)
        distances = (2.0 - 2.0 * similarity).clamp_min_(0.0)
        if exclude_diagonal:
            if query_count != reference_count:
                raise ValueError("Diagonal exclusion requires equal set sizes")
            diagonal = torch.arange(query_count, device=device)
            distances[diagonal, :, diagonal] = torch.inf
        output[:, start:stop] = distances.min(dim=-1).values.cpu()

    return output.reshape(query_count, height, width)


def robust_scores(train_distances: Tensor, test_distances: Tensor) -> tuple[Tensor, Tensor]:
    flat_train = train_distances.flatten(1)
    flat_test = test_distances.flatten(1)
    location_median = flat_train.median(dim=0).values
    location_mad = (flat_train - location_median).abs().median(dim=0).values
    # A floor prevents unusually invariant locations from dominating numerically.
    scale_floor = torch.quantile(location_mad, 0.25).clamp_min(1e-4)
    robust_scale = torch.maximum(1.4826 * location_mad, scale_floor)
    standardized = ((flat_test - location_median) / robust_scale).clamp_min(0.0)

    top_k = max(1, math.ceil(standardized.shape[1] * 0.01))
    anomaly_score = standardized.topk(top_k, dim=1).values.mean(dim=1)
    raw_score = flat_test.topk(top_k, dim=1).values.mean(dim=1)
    return anomaly_score, raw_score


def write_scores(
    output_path: Path, names: list[str], scores: Tensor, raw_scores: Tensor
) -> None:
    rows = sorted(
        zip(names, scores.tolist(), raw_scores.tolist()),
        key=lambda row: row[0],
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filename", "anomaly_score", "raw_patch_distance"))
        for name, score, raw_score in rows:
            writer.writerow((name, f"{score:.8f}", f"{raw_score:.8f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_paths = image_files(args.train_dir)
    test_paths = image_files(args.test_dir)
    if not train_paths or not test_paths:
        raise RuntimeError("Both train and test directories must contain images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    transform = weights.transforms(crop_size=256, resize_size=256)
    model = FeatureExtractor().eval().to(device)

    print(f"Extracting {len(train_paths)} train and {len(test_paths)} test images on {device}")
    train_features, _ = extract_features(
        train_paths, model, transform, device, args.batch_size
    )
    test_features, test_names = extract_features(
        test_paths, model, transform, device, args.batch_size
    )
    del model
    train_descriptors, test_descriptors = project_and_normalize(
        train_features, test_features, args.projection_dim, args.seed
    )
    del train_features, test_features

    print("Calibrating normal patch distances")
    train_distances = nearest_distances(
        train_descriptors, train_descriptors, device, exclude_diagonal=True
    )
    print("Scoring test patch distances")
    test_distances = nearest_distances(test_descriptors, train_descriptors, device)
    scores, raw_scores = robust_scores(train_distances, test_distances)
    write_scores(args.output, test_names, scores, raw_scores)
    print(f"Wrote {len(test_names)} scores to {args.output}")


if __name__ == "__main__":
    main()

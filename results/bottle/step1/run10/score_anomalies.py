#!/usr/bin/env python3
"""Score test images with PaDiM using only normal training images."""

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
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_SUFFIXES
    ]


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform: nn.Module) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.paths[index].name


class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)

    def forward(self, x: Tensor) -> Tensor:
        model = self.backbone
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        layer1 = model.layer1(x)
        layer2 = model.layer2(layer1)
        layer3 = model.layer3(layer2)

        # A 32x32 spatial grid preserves small defects while keeping covariance
        # estimation tractable. Pooling also reduces sensitivity to one-pixel shifts.
        size = (32, 32)
        features = torch.cat(
            [
                F.adaptive_avg_pool2d(layer1, size),
                F.interpolate(layer2, size=size, mode="bilinear", align_corners=False),
                F.interpolate(layer3, size=size, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        return features


@torch.inference_mode()
def extract_features(
    extractor: nn.Module,
    paths: list[Path],
    transform: nn.Module,
    channel_indices: Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[Tensor, list[str]]:
    loader = DataLoader(
        ImageDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, max(1, batch_size)),
        pin_memory=device.type == "cuda",
    )
    batches: list[Tensor] = []
    names: list[str] = []
    for images, batch_names in loader:
        features = extractor(images.to(device, non_blocking=True))
        features = features.index_select(1, channel_indices)
        batches.append(features.cpu())
        names.extend(batch_names)
    return torch.cat(batches), names


def fit_padim(features: Tensor, regularization: float) -> tuple[Tensor, Tensor]:
    # Arrange as [spatial_position, sample, feature_channel].
    samples = features.permute(2, 3, 0, 1).reshape(-1, features.shape[0], features.shape[1])
    mean = samples.mean(dim=1)
    centered = samples - mean[:, None, :]
    covariance = torch.bmm(centered.transpose(1, 2), centered)
    covariance /= max(1, samples.shape[1] - 1)
    identity = torch.eye(samples.shape[2], dtype=samples.dtype)[None]
    covariance += regularization * identity

    # Cholesky factors are smaller and more numerically stable than saved inverses.
    cholesky, info = torch.linalg.cholesky_ex(covariance)
    if torch.any(info):
        failed = int(torch.count_nonzero(info))
        raise RuntimeError(
            f"Covariance factorization failed at {failed} positions; "
            "increase --regularization."
        )
    return mean, cholesky


def anomaly_maps(features: Tensor, mean: Tensor, cholesky: Tensor) -> Tensor:
    samples = features.permute(2, 3, 0, 1).reshape(mean.shape[0], features.shape[0], -1)
    delta = samples - mean[:, None, :]
    solved = torch.linalg.solve_triangular(cholesky, delta.transpose(1, 2), upper=False)
    distances = torch.sqrt(torch.sum(solved.square(), dim=1).clamp_min(0))
    return distances.transpose(0, 1).reshape(features.shape[0], features.shape[2], features.shape[3])


def image_scores(maps: Tensor, top_fraction: float) -> Tensor:
    flat = maps.flatten(1)
    count = max(1, int(round(flat.shape[1] * top_fraction)))
    return flat.topk(count, dim=1).values.mean(dim=1)


def robust_z(values: Tensor, center: float, scale: float) -> Tensor:
    return (values - center) / max(scale, 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--model-output", type=Path, default=Path("padim_model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths = image_files(args.train_dir)
    test_paths = image_files(args.test_dir)
    if not train_paths:
        raise RuntimeError(f"No training images found in {args.train_dir}")
    if not test_paths:
        raise RuntimeError(f"No test images found in {args.test_dir}")

    weights = Wide_ResNet50_2_Weights.DEFAULT
    extractor = FeatureExtractor().eval().to(device)
    total_channels = 256 + 512 + 1024
    generator = torch.Generator().manual_seed(args.seed)
    channel_indices = torch.randperm(total_channels, generator=generator)[: args.feature_dim]
    channel_indices_device = channel_indices.to(device)

    train_features, _ = extract_features(
        extractor,
        train_paths,
        weights.transforms(),
        channel_indices_device,
        device,
        args.batch_size,
    )
    mean, cholesky = fit_padim(train_features.float(), args.regularization)
    train_raw = image_scores(
        anomaly_maps(train_features.float(), mean, cholesky), args.top_fraction
    )
    median = float(train_raw.median())
    mad = float((train_raw - median).abs().median())
    robust_scale = 1.4826 * mad

    test_features, names = extract_features(
        extractor,
        test_paths,
        weights.transforms(),
        channel_indices_device,
        device,
        args.batch_size,
    )
    test_maps = anomaly_maps(test_features.float(), mean, cholesky)
    raw_scores = image_scores(test_maps, args.top_fraction)
    scores = robust_z(raw_scores, median, robust_scale)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["image", "anomaly_score", "raw_score"])
        for name, score, raw_score in zip(names, scores.tolist(), raw_scores.tolist()):
            writer.writerow([name, f"{score:.8f}", f"{raw_score:.8f}"])

    artifact = {
        "method": "PaDiM",
        "backbone": "ImageNet-pretrained Wide ResNet-50-2",
        "channel_indices": channel_indices,
        "mean": mean,
        "cholesky": cholesky,
        "train_score_median": median,
        "train_score_robust_scale": robust_scale,
        "top_fraction": args.top_fraction,
        "regularization": args.regularization,
    }
    torch.save(artifact, args.model_output)

    summary = {
        "method": artifact["method"],
        "backbone": artifact["backbone"],
        "device": str(device),
        "training_images": len(train_paths),
        "test_images": len(test_paths),
        "output": str(args.output),
        "model_output": str(args.model_output),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

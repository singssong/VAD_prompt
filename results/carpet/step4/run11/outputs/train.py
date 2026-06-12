#!/usr/bin/env python3
"""Train a feature-distribution anomaly detector using normal images only."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


@dataclass(frozen=True)
class Config:
    train_dir: str = "./data/train"
    test_dir: str = "./data/test_images"
    output_dir: str = "./outputs"
    model_path: str = "./outputs/model.pt"
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 4
    selected_channels: int = 64
    covariance_regularization: float = 0.01
    smoothing_sigma: float = 4.0
    image_top_fraction: float = 0.01
    seed: int = 17


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_paths(folder: str | Path) -> list[Path]:
    paths = sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No images found in {folder}")
    return paths


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int):
        self.paths = paths
        self.image_size = image_size
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        self.mean = torch.tensor(weights.transforms().mean).view(3, 1, 1)
        self.std = torch.tensor(weights.transforms().std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(array)
        return (tensor - self.mean) / self.std, path.name


class FeatureExtractor(nn.Module):
    """Return concatenated, spatially aligned layer2/layer3 feature maps."""

    def __init__(self) -> None:
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat((layer2, layer3), dim=1)


def make_loader(paths: list[Path], config: Config, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ImageDataset(paths, config.image_size),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


@torch.inference_mode()
def extract_features(
    extractor: nn.Module,
    loader: DataLoader,
    device: torch.device,
    selected_indices: torch.Tensor,
) -> tuple[torch.Tensor, list[str]]:
    batches: list[torch.Tensor] = []
    names: list[str] = []
    selected_indices = selected_indices.to(device)
    for images, batch_names in loader:
        features = extractor(images.to(device, non_blocking=True))
        features = features.index_select(1, selected_indices)
        batches.append(features.cpu())
        names.extend(batch_names)
    return torch.cat(batches), names


def fit_normal_model(
    features: torch.Tensor, regularization: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one regularized Gaussian at every feature-map location."""
    samples = features.permute(0, 2, 3, 1).float()
    mean = samples.mean(dim=0)
    centered = samples - mean
    covariance = torch.einsum("nhwc,nhwd->hwcd", centered, centered)
    covariance /= max(samples.shape[0] - 1, 1)
    channels = samples.shape[-1]
    eye = torch.eye(channels, dtype=covariance.dtype).view(1, 1, channels, channels)
    covariance = covariance + regularization * eye
    precision = torch.linalg.inv(covariance)
    return mean.cpu(), precision.cpu()


def gaussian_kernel2d(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, math.ceil(3.0 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel1d = torch.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
    kernel1d /= kernel1d.sum()
    return torch.outer(kernel1d, kernel1d).view(1, 1, 2 * radius + 1, 2 * radius + 1)


def smooth_maps(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel2d(sigma, maps.device)
    padding = kernel.shape[-1] // 2
    padded = F.pad(maps.unsqueeze(1), (padding,) * 4, mode="reflect")
    return F.conv2d(padded, kernel).squeeze(1)


def score_feature_maps(
    features: torch.Tensor,
    mean: torch.Tensor,
    precision: torch.Tensor,
    smoothing_sigma: float,
    top_fraction: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute smoothed pixel maps and top-tail aggregate image scores."""
    all_maps: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    mean = mean.to(device)
    precision = precision.to(device)
    for batch in features.split(16):
        batch = batch.to(device).permute(0, 2, 3, 1).float()
        delta = batch - mean
        squared = torch.einsum("nhwc,hwcd,nhwd->nhw", delta, precision, delta)
        maps = smooth_maps(torch.sqrt(squared.clamp_min(0.0)), smoothing_sigma)
        flat = maps.flatten(1)
        top_count = max(1, round(flat.shape[1] * top_fraction))
        scores = flat.topk(top_count, dim=1).values.mean(dim=1)
        all_maps.append(maps.cpu())
        all_scores.append(scores.cpu())
    return torch.cat(all_maps), torch.cat(all_scores)


def robust_range(values: torch.Tensor) -> tuple[float, float]:
    low = float(torch.quantile(values.float(), 0.50))
    high = float(torch.quantile(values.float(), 0.995))
    if high <= low:
        high = low + 1e-6
    return low, high


def train(config: Config) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = image_paths(config.train_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor().to(device)

    total_channels = 512 + 1024
    generator = torch.Generator().manual_seed(config.seed)
    selected_indices = torch.randperm(total_channels, generator=generator)[
        : config.selected_channels
    ]
    features, _ = extract_features(
        extractor, make_loader(paths, config), device, selected_indices
    )
    mean, precision = fit_normal_model(features, config.covariance_regularization)
    train_maps, train_scores = score_feature_maps(
        features,
        mean,
        precision,
        config.smoothing_sigma,
        config.image_top_fraction,
        device,
    )
    image_low, image_high = robust_range(train_scores)
    pixel_low, pixel_high = robust_range(train_maps.flatten())

    artifact = {
        "config": asdict(config),
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "selected_indices": selected_indices,
        "mean": mean,
        "precision": precision,
        "image_score_range": [image_low, image_high],
        "pixel_score_range": [pixel_low, pixel_high],
    }
    torch.save(artifact, config.model_path)
    print(f"Trained on {len(paths)} normal images using {device}.")
    print(f"Saved model to {config.model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default=Config.train_dir)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--model-path", default=Config.model_path)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        Config(
            train_dir=args.train_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )

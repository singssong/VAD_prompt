from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from config import Config


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, paths: Iterable[Path], image_size: int):
        self.paths = list(paths)
        self.image_size = image_size
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return tensor, path.name


class FeatureExtractor(nn.Module):
    """Wide ResNet feature extractor returning aligned layer2/layer3 patches."""

    def __init__(self):
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
        features = self.stem(images)
        features = self.layer1(features)
        layer2 = self.layer2(features)
        layer3 = self.layer3(layer2)

        # Local averaging improves robustness while retaining the 32x32 patch grid.
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        layer2 = F.normalize(layer2, dim=1)
        layer3 = F.normalize(layer3, dim=1)
        return torch.cat((layer2, layer3), dim=1)


def create_projection(
    input_dim: int, output_dim: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    projection /= math.sqrt(output_dim)
    return projection.to(device)


@torch.inference_mode()
def extract_features(
    extractor: FeatureExtractor,
    images: torch.Tensor,
    projection: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Extract and concatenate two mid-level maps, then project patch vectors."""
    feature_map = extractor(images)
    batch, channels, height, width = feature_map.shape
    patches = feature_map.permute(0, 2, 3, 1).reshape(-1, channels)
    patches = patches @ projection
    patches = F.normalize(patches, dim=1)
    return patches.reshape(batch, height * width, -1), (height, width)


def build_normal_model(
    patch_batches: list[torch.Tensor], memory_bank_size: int, seed: int
) -> torch.Tensor:
    """Store a reproducible random coreset of normal patch embeddings."""
    patches = torch.cat(patch_batches, dim=0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = min(memory_bank_size, len(patches))
    indices = torch.randperm(len(patches), generator=generator)[:count]
    return patches[indices].contiguous()


def nearest_neighbor_distances(
    patches: torch.Tensor,
    memory_bank: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    minimum = torch.full(
        patches.shape[:-1], float("inf"), device=patches.device
    )
    flat = patches.reshape(-1, patches.shape[-1])
    flat_minimum = minimum.reshape(-1)
    for start in range(0, len(memory_bank), chunk_size):
        chunk = memory_bank[start : start + chunk_size]
        distances = torch.cdist(flat, chunk)
        flat_minimum = torch.minimum(flat_minimum, distances.min(dim=1).values)
    return flat_minimum.reshape(patches.shape[:-1])


def gaussian_smooth(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=maps.device)
    kernel = torch.exp(-(coordinates.float() ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    kernel_2d = torch.outer(kernel, kernel)[None, None]
    return F.conv2d(maps, kernel_2d, padding=radius)


def aggregate_image_scores(maps: torch.Tensor, top_fraction: float) -> torch.Tensor:
    flat = maps.flatten(1)
    count = max(1, int(math.ceil(flat.shape[1] * top_fraction)))
    return flat.topk(count, dim=1).values.mean(dim=1)


@torch.inference_mode()
def score_features(
    patches: torch.Tensor,
    grid_size: tuple[int, int],
    memory_bank: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return smoothed patch anomaly maps and top-tail image scores."""
    distances = nearest_neighbor_distances(
        patches, memory_bank, config.neighbor_chunk_size
    )
    maps = distances.reshape(-1, 1, *grid_size)
    maps = gaussian_smooth(maps, config.gaussian_sigma)
    image_scores = aggregate_image_scores(maps, config.top_fraction)
    return maps[:, 0], image_scores


def robust_range(values: torch.Tensor, low_q: float, high_q: float) -> tuple[float, float]:
    values = values.float().flatten()
    low = float(torch.quantile(values, low_q))
    high = float(torch.quantile(values, high_q))
    if high <= low:
        high = low + 1e-6
    return low, high


def normalize_scores(values: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return ((values - low) / (high - low)).clamp(0.0, 1.0)


def normalize_image_scores(
    values: torch.Tensor, low: float, high: float
) -> torch.Tensor:
    """Map scores to [0, 1) without saturating large anomaly distances."""
    shifted = (values - low).clamp_min(0.0)
    scale = max(high - low, 1e-6)
    return shifted / (shifted + scale)

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18

import config


class ImageDataset(Dataset):
    def __init__(self, paths: Iterable[Path]) -> None:
        self.paths = list(paths)
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms(
            crop_size=config.IMAGE_SIZE,
            resize_size=config.IMAGE_SIZE,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            # Resize explicitly as required, without the pretrained center-crop recipe.
            image = image.convert("RGB").resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            tensor = self.transform(image)
        return tensor, path.name


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )


class MultiScaleFeatureExtractor(nn.Module):
    """ImageNet ResNet-18 truncated at layer3, returning layer2 and layer3."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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
        level2 = self.layer2(features)
        level3 = self.layer3(level2)
        level3 = F.interpolate(
            level3, size=level2.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat((level2, level3), dim=1)


@torch.inference_mode()
def extract_features(
    extractor: nn.Module, images: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Extract concatenated mid-level patch features as [B, H, W, C]."""
    feature_map = extractor(images.to(device, non_blocking=True))
    return feature_map.permute(0, 2, 3, 1).contiguous()


def model_normal_features(
    feature_batches: list[torch.Tensor], memory_bank_size: int
) -> dict[str, torch.Tensor | dict[str, object]]:
    """Standardize normal patches and store a deterministic sampled memory bank."""
    all_features = torch.cat(feature_batches, dim=0).float()
    channel_mean = all_features.mean(dim=0)
    channel_std = all_features.std(dim=0).clamp_min(1e-6)
    standardized = (all_features - channel_mean) / channel_std

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    count = min(memory_bank_size, standardized.shape[0])
    indices = torch.randperm(standardized.shape[0], generator=generator)[:count]
    memory_bank = standardized[indices].contiguous()

    return {
        "memory_bank": memory_bank,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "metadata": {
            "backbone": config.BACKBONE,
            "feature_layers": config.FEATURE_LAYERS,
            "image_size": config.IMAGE_SIZE,
            "feature_grid_size": config.FEATURE_GRID_SIZE,
        },
    }


def nearest_neighbor_distances(
    queries: torch.Tensor,
    memory_bank: torch.Tensor,
    query_chunk: int,
    bank_chunk: int,
) -> torch.Tensor:
    """Compute exact nearest-neighbor distances with bounded GPU memory."""
    outputs = []
    for query_start in range(0, queries.shape[0], query_chunk):
        query = queries[query_start : query_start + query_chunk]
        nearest = torch.full(
            (query.shape[0],), float("inf"), device=query.device
        )
        for bank_start in range(0, memory_bank.shape[0], bank_chunk):
            bank = memory_bank[bank_start : bank_start + bank_chunk]
            nearest = torch.minimum(nearest, torch.cdist(query, bank).amin(dim=1))
        outputs.append(nearest)
    return torch.cat(outputs)


def gaussian_kernel2d(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, math.ceil(3 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)[None, None]


def smooth_and_resize_map(
    anomaly_map: torch.Tensor, sigma: float, output_size: int
) -> torch.Tensor:
    """Gaussian-smooth a patch map, then resize it to the required resolution."""
    kernel = gaussian_kernel2d(sigma, anomaly_map.device)
    padding = kernel.shape[-1] // 2
    smoothed = F.conv2d(anomaly_map[None, None], kernel, padding=padding)
    resized = F.interpolate(
        smoothed,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0]


@torch.inference_mode()
def score_feature_map(
    feature_map: torch.Tensor,
    model: dict[str, torch.Tensor | dict[str, object]],
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Score every patch and aggregate the largest one percent into an image score."""
    mean = model["channel_mean"].to(device)
    std = model["channel_std"].to(device)
    memory_bank = model["memory_bank"].to(device)
    height, width, channels = feature_map.shape
    queries = ((feature_map.to(device) - mean) / std).reshape(-1, channels)
    distances = nearest_neighbor_distances(
        queries,
        memory_bank,
        config.DISTANCE_QUERY_CHUNK,
        config.DISTANCE_BANK_CHUNK,
    )
    patch_map = distances.reshape(height, width)
    pixel_map = smooth_and_resize_map(
        patch_map, config.GAUSSIAN_SIGMA, config.IMAGE_SIZE
    )
    top_count = max(1, round(pixel_map.numel() * config.IMAGE_SCORE_TOP_FRACTION))
    image_score = float(torch.topk(pixel_map.flatten(), top_count).values.mean())
    return pixel_map.cpu(), image_score


def robust_normalize(values: torch.Tensor) -> torch.Tensor:
    """Map values consistently to [0, 1] using robust 1st/99th percentiles."""
    flat = values.float().flatten()
    low = torch.quantile(flat, 0.01)
    high = torch.quantile(flat, 0.99)
    if float(high - low) < 1e-12:
        return torch.zeros_like(values, dtype=torch.float32)
    return ((values.float() - low) / (high - low)).clamp(0.0, 1.0)

import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

import config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, paths: Iterable[Path]):
        self.paths = list(paths)
        self.transform = Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms(
            crop_size=config.IMAGE_SIZE,
            resize_size=config.IMAGE_SIZE,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            tensor = self.transform(image)
        return tensor, path.name


class FeatureExtractor(nn.Module):
    """Extract and concatenate aligned mid-level Wide ResNet patch features."""

    def __init__(self):
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.requires_grad_(False)
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = self.layer1(x)
        level2 = self.layer2(x)
        level3 = self.layer3(level2)
        level3 = F.interpolate(
            level3, size=level2.shape[-2:], mode="bilinear", align_corners=False
        )
        features = torch.cat([level2, level3], dim=1)
        return F.normalize(features, dim=1)


def make_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    projection /= math.sqrt(output_dim)
    return projection


def flatten_and_project(feature_map: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    patches = feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])
    patches = patches @ projection.to(patches.device)
    return F.normalize(patches, dim=1)


def build_normal_model(
    patch_features: torch.Tensor,
    projection: torch.Tensor,
    image_scores: np.ndarray,
    pixel_scores: np.ndarray,
) -> dict:
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    count = min(config.MEMORY_BANK_SIZE, len(patch_features))
    indices = torch.randperm(len(patch_features), generator=generator)[:count]
    memory_bank = patch_features[indices].contiguous().half()

    image_low = float(np.quantile(image_scores, 0.50))
    image_high = float(np.quantile(image_scores, 0.995))
    if image_high <= image_low:
        image_high = image_low + 1e-6
    pixel_high = float(np.quantile(pixel_scores, 0.995))
    pixel_high = max(pixel_high, 1e-6)

    return {
        "memory_bank": memory_bank,
        "projection": projection.half(),
        "image_low": image_low,
        "image_high": image_high,
        "pixel_high": pixel_high,
        "config": {
            "backbone": config.BACKBONE,
            "feature_layers": config.FEATURE_LAYERS,
            "image_size": config.IMAGE_SIZE,
            "feature_grid_size": config.FEATURE_GRID_SIZE,
            "projection_dim": config.PROJECTION_DIM,
        },
    }


def nearest_neighbor_distances(
    queries: torch.Tensor, memory_bank: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    memory_bank = memory_bank.to(queries.device, dtype=torch.float32)
    distances = []
    for start in range(0, len(queries), chunk_size):
        query = queries[start : start + chunk_size].float()
        # Unit vectors make squared Euclidean distance equal to 2 - 2*cosine.
        similarity = query @ memory_bank.T
        nearest = similarity.max(dim=1).values
        distances.append(torch.sqrt(torch.clamp(2.0 - 2.0 * nearest, min=0.0)))
    return torch.cat(distances)


def gaussian_smooth(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=maps.device, dtype=maps.dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    maps = F.pad(maps, (radius, radius, 0, 0), mode="reflect")
    maps = F.conv2d(maps, kernel.view(1, 1, 1, -1))
    maps = F.pad(maps, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(maps, kernel.view(1, 1, -1, 1))


def score_feature_maps(
    feature_maps: torch.Tensor,
    projection: torch.Tensor,
    memory_bank: torch.Tensor,
) -> torch.Tensor:
    batch_size, _, height, width = feature_maps.shape
    patches = flatten_and_project(feature_maps, projection)
    distances = nearest_neighbor_distances(
        patches, memory_bank, config.NN_QUERY_CHUNK
    )
    maps = distances.reshape(batch_size, 1, height, width)
    return gaussian_smooth(maps, config.GAUSSIAN_SIGMA)


def aggregate_image_scores(anomaly_maps: torch.Tensor) -> torch.Tensor:
    flat = anomaly_maps.flatten(1)
    top_count = max(1, int(flat.shape[1] * config.IMAGE_TOP_FRACTION))
    return flat.topk(top_count, dim=1).values.mean(dim=1)


def normalize_image_score(raw_score: float, low: float, high: float) -> float:
    # A logistic calibration keeps scores bounded without collapsing all strong
    # anomalies to exactly 1. The normal median maps to ~0.27 and q99.5 to 0.5.
    z = np.clip((raw_score - high) / (high - low), -30.0, 30.0)
    return float(1.0 / (1.0 + np.exp(-z)))


def save_pixel_map(anomaly_map: torch.Tensor, path: Path, pixel_high: float) -> None:
    resized = F.interpolate(
        anomaly_map[None, None],
        size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    normalized = torch.clamp(resized / pixel_high, 0.0, 1.0)
    array = (normalized.cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)

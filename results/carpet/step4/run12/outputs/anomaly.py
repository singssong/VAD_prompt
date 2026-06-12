from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import config


class ImageDataset(Dataset):
    def __init__(self, directory: Path):
        self.paths = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No supported images found in {directory}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image,
                [config.IMAGE_SIZE, config.IMAGE_SIZE],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.to_tensor(image)
        tensor = TF.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return tensor, path.name


class MidLevelResNet18(nn.Module):
    """Frozen ImageNet backbone returning layer2 and layer3 activations."""

    def __init__(self) -> None:
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool, model.layer1
        )
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.requires_grad_(False)
        self.eval()

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        x = self.stem(images)
        level2 = self.layer2(x)
        level3 = self.layer3(level2)
        return level2, level3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_feature_map(backbone: nn.Module, images: Tensor) -> Tensor:
    """Concatenate aligned mid-level features into BxCxHxW descriptors."""
    level2, level3 = backbone(images)
    level3 = F.interpolate(
        level3, size=level2.shape[-2:], mode="bilinear", align_corners=False
    )
    features = torch.cat([level2, level3], dim=1)
    return F.normalize(features, p=2, dim=1)


@torch.inference_mode()
def build_normal_memory(
    backbone: nn.Module,
    batches: Iterable[tuple[Tensor, list[str]]],
    device: torch.device,
) -> Tensor:
    """Extract normal patch descriptors and retain a bounded random coreset."""
    sampled_batches: list[Tensor] = []
    generator = torch.Generator().manual_seed(config.SEED)

    for images, _ in batches:
        feature_map = extract_feature_map(backbone, images.to(device))
        patches = feature_map.permute(0, 2, 3, 1).reshape(
            feature_map.shape[0], -1, feature_map.shape[1]
        )
        count = min(config.PATCHES_PER_IMAGE, patches.shape[1])
        indices = torch.randperm(
            patches.shape[1], generator=generator, device="cpu"
        )[:count].to(device)
        sampled_batches.append(patches[:, indices].reshape(-1, patches.shape[-1]).cpu())

    memory = torch.cat(sampled_batches, dim=0)
    if len(memory) > config.MEMORY_BANK_SIZE:
        indices = torch.randperm(len(memory), generator=generator)[: config.MEMORY_BANK_SIZE]
        memory = memory[indices]
    return memory.contiguous()


@torch.inference_mode()
def score_feature_map(feature_map: Tensor, memory: Tensor) -> Tensor:
    """Return a nearest-normal-patch anomaly map for each input image."""
    batch, channels, height, width = feature_map.shape
    queries = feature_map.permute(0, 2, 3, 1).reshape(-1, channels)
    distances: list[Tensor] = []
    for start in range(0, len(queries), config.DISTANCE_QUERY_CHUNK):
        chunk = queries[start : start + config.DISTANCE_QUERY_CHUNK]
        distances.append(torch.cdist(chunk, memory).amin(dim=1))
    return torch.cat(distances).reshape(batch, height, width)


def gaussian_smooth(maps: Tensor, sigma: float) -> Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    coordinates = torch.arange(
        -radius, radius + 1, device=maps.device, dtype=maps.dtype
    )
    kernel = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel2d = torch.outer(kernel, kernel)[None, None]
    padded = F.pad(
        maps[:, None], (radius, radius, radius, radius), mode="reflect"
    )
    return F.conv2d(padded, kernel2d)[:, 0]


def robust_normalize(
    values: np.ndarray, low_percentile: float, high_percentile: float
) -> np.ndarray:
    low, high = np.percentile(values, [low_percentile, high_percentile])
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def aggregate_image_score(pixel_map: np.ndarray) -> float:
    flat = pixel_map.reshape(-1)
    count = max(1, int(round(flat.size * config.IMAGE_TOP_FRACTION)))
    return float(np.partition(flat, flat.size - count)[-count:].mean())

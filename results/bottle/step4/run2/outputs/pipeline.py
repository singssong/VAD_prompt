from __future__ import annotations

import math
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

import config


def image_files(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.transform = v2.Compose(
            [
                v2.Resize(
                    (config.IMAGE_SIZE, config.IMAGE_SIZE),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.paths[index].name


class FeatureExtractor(nn.Module):
    """ImageNet backbone exposing two mid-level feature maps."""

    def __init__(self):
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
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
        level2 = self.layer2(x)
        level3 = self.layer3(level2)
        level3 = F.interpolate(
            level3, size=level2.shape[-2:], mode="bilinear", align_corners=False
        )
        features = torch.cat((level2, level3), dim=1)
        # Local averaging gives each patch a useful receptive-field neighborhood.
        return F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)


def make_loader(paths: list[Path], shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ImageDataset(paths),
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def make_projection(input_dim: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(config.SEED)
    projection = torch.randn(
        input_dim, config.PROJECTION_DIM, generator=generator
    ) / math.sqrt(config.PROJECTION_DIM)
    return projection.to(device)


def extract_features(
    extractor: FeatureExtractor,
    images: torch.Tensor,
    projection: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Extract, concatenate, project, and L2-normalize spatial patch features."""
    feature_map = extractor(images)
    batch, channels, height, width = feature_map.shape
    patches = feature_map.permute(0, 2, 3, 1).reshape(-1, channels)
    patches = patches @ projection
    patches = F.normalize(patches, dim=1)
    return patches.reshape(batch, height * width, -1), (height, width)


def build_normal_model(
    extractor: FeatureExtractor,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Create a bounded random memory bank representing normal patch features."""
    projection = None
    reservoir = torch.empty(
        (config.MEMORY_SIZE, config.PROJECTION_DIM), dtype=torch.float32
    )
    seen = 0
    stored = 0
    rng = random.Random(config.SEED)

    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            if projection is None:
                raw = extractor(images)
                projection = make_projection(raw.shape[1], device)
                batch, channels, height, width = raw.shape
                patches = raw.permute(0, 2, 3, 1).reshape(-1, channels)
                patches = F.normalize(patches @ projection, dim=1).cpu()
            else:
                patches, _ = extract_features(extractor, images, projection)
                patches = patches.reshape(-1, config.PROJECTION_DIM).cpu()

            for patch in patches:
                seen += 1
                if stored < config.MEMORY_SIZE:
                    reservoir[stored] = patch
                    stored += 1
                else:
                    replacement = rng.randrange(seen)
                    if replacement < config.MEMORY_SIZE:
                        reservoir[replacement] = patch

    if projection is None or stored == 0:
        raise RuntimeError("No training images were found.")
    return {
        "projection": projection.cpu(),
        "memory": reservoir[:stored],
        "feature_grid": (height, width),
    }


def nearest_distances(
    queries: torch.Tensor, memory: torch.Tensor, chunk_size: int = 2048
) -> torch.Tensor:
    outputs = []
    for chunk in queries.split(chunk_size):
        # Unit-normalized vectors: squared Euclidean distance is 2 - 2*cosine.
        similarities = chunk @ memory.T
        outputs.append(torch.sqrt((2.0 - 2.0 * similarities.max(dim=1).values).clamp_min(0)))
    return torch.cat(outputs)


def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)[None, None]


def score_batch(
    extractor: FeatureExtractor,
    images: torch.Tensor,
    model: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return smoothed 256x256 anomaly maps and top-tail image scores."""
    projection = model["projection"].to(device)
    memory = model["memory"].to(device)
    patches, (height, width) = extract_features(extractor, images, projection)
    distances = nearest_distances(
        patches.reshape(-1, patches.shape[-1]), memory
    ).reshape(images.shape[0], 1, height, width)

    kernel = gaussian_kernel(config.GAUSSIAN_SIGMA, device)
    padding = kernel.shape[-1] // 2
    smoothed = F.conv2d(distances, kernel, padding=padding)
    maps = F.interpolate(
        smoothed,
        size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    count = max(1, int(maps[0].numel() * config.IMAGE_TOP_FRACTION))
    image_scores = maps.flatten(1).topk(count, dim=1).values.mean(dim=1)
    return maps, image_scores


def fit_score_calibration(
    extractor: FeatureExtractor,
    loader: DataLoader,
    model: dict,
    device: torch.device,
) -> dict:
    map_values = []
    image_values = []
    with torch.inference_mode():
        for images, _ in loader:
            maps, scores = score_batch(
                extractor, images.to(device, non_blocking=True), model, device
            )
            # Sampling is enough for stable map quantiles and keeps memory bounded.
            map_values.append(maps[:, ::4, ::4].flatten().cpu())
            image_values.append(scores.cpu())
    pixels = torch.cat(map_values)
    images = torch.cat(image_values)
    return {
        "pixel_low": float(torch.quantile(pixels, 0.01)),
        "pixel_high": float(torch.quantile(pixels, 0.995)),
        "image_low": float(torch.quantile(images, 0.01)),
        "image_high": float(torch.quantile(images, 0.99)),
    }


def normalize(values: torch.Tensor, low: float, high: float) -> torch.Tensor:
    scale = max(high - low, 1e-8)
    return ((values - low) / scale).clamp(0.0, 1.0)


def normalize_image_scores(
    values: torch.Tensor, low: float, high: float
) -> torch.Tensor:
    """Map scores monotonically to (0, 1) without clipping anomalous rankings."""
    del low
    scale = max(high, 1e-8)
    positive = values.clamp_min(0.0)
    return positive / (positive + scale)


def save_pixel_map(path: Path, normalized_map: torch.Tensor) -> None:
    array = (normalized_map.cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)

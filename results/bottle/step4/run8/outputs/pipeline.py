from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import config


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
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
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        return tensor, path.name


class FeatureExtractor(nn.Module):
    """ImageNet Wide ResNet-50-2 truncated at two mid-level stages."""

    def __init__(self) -> None:
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool, model.layer1
        )
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stem(images)
        level2 = self.layer2(features)
        level3 = self.layer3(level2)
        return level2, level3


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    projection /= math.sqrt(output_dim)
    return projection


@torch.inference_mode()
def extract_features(
    extractor: nn.Module,
    images: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Return a BxHxWxD grid made from two concatenated feature depths."""
    level2, level3 = extractor(images)
    level2 = F.avg_pool2d(level2, kernel_size=3, stride=1, padding=1)
    level3 = F.avg_pool2d(level3, kernel_size=3, stride=1, padding=1)
    level3 = F.interpolate(
        level3, size=level2.shape[-2:], mode="bilinear", align_corners=False
    )
    level2 = F.normalize(level2, dim=1)
    level3 = F.normalize(level3, dim=1)
    combined = torch.cat((level2, level3), dim=1).permute(0, 2, 3, 1)
    projected = combined @ projection
    return F.normalize(projected, dim=-1)


def build_normal_model(
    extractor: nn.Module,
    train_paths: list[Path],
    device: torch.device,
) -> dict[str, object]:
    """Build a normal patch memory and calibrate it with held-out normal images."""
    generator = torch.Generator().manual_seed(config.SEED)
    order = torch.randperm(len(train_paths), generator=generator).tolist()
    calibration_count = max(1, round(len(train_paths) * config.CALIBRATION_FRACTION))
    calibration_paths = [train_paths[i] for i in order[:calibration_count]]
    reference_paths = [train_paths[i] for i in order[calibration_count:]]

    sample = ImageDataset(reference_paths)[0][0].unsqueeze(0).to(device)
    level2, level3 = extractor(sample)
    input_dim = level2.shape[1] + level3.shape[1]
    projection = create_projection(input_dim, config.PROJECTION_DIM, config.SEED)
    projection = projection.to(device)

    reference_loader = DataLoader(
        ImageDataset(reference_paths),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    all_patches = []
    for images, _ in reference_loader:
        features = extract_features(extractor, images.to(device), projection)
        all_patches.append(features.flatten(0, 2).cpu())
    all_patches = torch.cat(all_patches)
    memory_indices = torch.randperm(
        len(all_patches), generator=generator
    )[: config.MEMORY_BANK_SIZE]
    memory_bank = all_patches[memory_indices].contiguous()

    calibration_loader = DataLoader(
        ImageDataset(calibration_paths),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    calibration_maps = []
    memory_device = memory_bank.to(device)
    for images, _ in calibration_loader:
        features = extract_features(extractor, images.to(device), projection)
        calibration_maps.append(nearest_neighbor_maps(features, memory_device).cpu())
    calibration_maps = torch.cat(calibration_maps)
    calibration_scores = aggregate_image_scores(calibration_maps)

    image_low = torch.quantile(calibration_scores, 0.05).item()
    image_high = torch.quantile(calibration_scores, 0.99).item()
    pixel_low = torch.quantile(calibration_maps, 0.50).item()
    pixel_high = torch.quantile(calibration_maps, 0.995).item()
    epsilon = 1e-6
    image_high = max(image_high, image_low + epsilon)
    pixel_high = max(pixel_high, pixel_low + epsilon)

    return {
        "backbone": config.BACKBONE,
        "feature_layers": config.FEATURE_LAYERS,
        "image_size": config.IMAGE_SIZE,
        "projection": projection.cpu(),
        "memory_bank": memory_bank,
        "image_score_low": image_low,
        "image_score_high": image_high,
        "pixel_score_low": pixel_low,
        "pixel_score_high": pixel_high,
        "top_fraction": config.TOP_FRACTION,
        "reference_images": len(reference_paths),
        "calibration_images": len(calibration_paths),
    }


def nearest_neighbor_maps(
    features: torch.Tensor,
    memory_bank: torch.Tensor,
) -> torch.Tensor:
    """Compute one nearest-normal-patch distance for every spatial feature."""
    batch, height, width, channels = features.shape
    queries = features.reshape(-1, channels)
    query_norm = (queries * queries).sum(dim=1, keepdim=True)
    memory_norm = (memory_bank * memory_bank).sum(dim=1).unsqueeze(0)
    squared_distances = query_norm + memory_norm - 2.0 * (queries @ memory_bank.T)
    distances = squared_distances.clamp_min_(0).min(dim=1).values.sqrt_()
    return distances.reshape(batch, height, width)


def aggregate_image_scores(anomaly_maps: torch.Tensor) -> torch.Tensor:
    flat = anomaly_maps.flatten(1)
    count = max(1, math.ceil(flat.shape[1] * config.TOP_FRACTION))
    return flat.topk(count, dim=1).values.mean(dim=1)


def normalize_scores(
    values: torch.Tensor, low: float, high: float
) -> torch.Tensor:
    # Map the normal calibration interval from 0.01 to 0.5. A rational upper
    # tail approaches 1 without hard clipping, preserving anomalous rankings.
    position = (values - low) / (high - low)
    positive = position.clamp_min(0)
    denominator_offset = 0.5 / 0.49
    numerator_offset = 0.01 * denominator_offset
    upper = (positive + numerator_offset) / (
        positive + denominator_offset
    )
    lower = 0.01 * torch.exp(math.log(100.0) * position.clamp_max(0))
    return torch.where(position >= 0, upper, lower)


def postprocess_pixel_maps(
    anomaly_maps: torch.Tensor,
    low: float,
    high: float,
) -> torch.Tensor:
    """Gaussian smooth at feature resolution, then resize to the required size."""
    maps = anomaly_maps.unsqueeze(1)
    maps = TF.gaussian_blur(
        maps,
        kernel_size=[config.GAUSSIAN_KERNEL_SIZE, config.GAUSSIAN_KERNEL_SIZE],
        sigma=[config.GAUSSIAN_SIGMA, config.GAUSSIAN_SIGMA],
    )
    maps = F.interpolate(
        maps,
        size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    return normalize_scores(maps.squeeze(1), low, high)


def save_grayscale_png(score_map: torch.Tensor, path: Path) -> None:
    pixels = (score_map.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def make_loader(paths: Iterable[Path], device: torch.device) -> DataLoader:
    return DataLoader(
        ImageDataset(list(paths)),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

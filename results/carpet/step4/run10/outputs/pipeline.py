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
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from config import Config


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_files(directory: Path) -> list[Path]:
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise RuntimeError(f"No supported images found in {directory}")
    return files


class ImageDataset(Dataset):
    def __init__(self, files: Iterable[Path], image_size: int):
        self.files = list(files)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.files[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
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


def make_loader(
    files: Iterable[Path],
    config: Config,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        ImageDataset(files, config.image_size),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def build_feature_extractor(
    config: Config,
    device: torch.device,
) -> nn.Module:
    backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
    return_nodes = {layer: layer for layer in config.feature_layers}
    extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
    return extractor.eval().to(device)


def make_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    projection /= math.sqrt(output_dim)
    return projection


@torch.inference_mode()
def extract_features(
    extractor: nn.Module,
    images: torch.Tensor,
    projection: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    """Return L2-normalized patch embeddings with shape [B, H*W, D]."""
    feature_maps = extractor(images)
    resized = [
        F.interpolate(
            feature_maps[layer],
            size=(config.feature_grid_size, config.feature_grid_size),
            mode="bilinear",
            align_corners=False,
        )
        for layer in config.feature_layers
    ]
    features = torch.cat(resized, dim=1)
    patches = features.permute(0, 2, 3, 1).reshape(features.shape[0], -1, features.shape[1])
    projected = patches @ projection
    return F.normalize(projected, dim=-1)


@torch.inference_mode()
def build_normal_feature_model(
    extractor: nn.Module,
    loader: DataLoader,
    projection: torch.Tensor,
    config: Config,
    device: torch.device,
) -> torch.Tensor:
    """Extract normal patches and store a deterministic uniform subset."""
    chunks: list[torch.Tensor] = []
    for batch_index, (images, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        embeddings = extract_features(extractor, images, projection, config)
        chunks.append(embeddings.reshape(-1, embeddings.shape[-1]).cpu())
        print(f"\rExtracting normal features: batch {batch_index}/{len(loader)}", end="", flush=True)
    print()

    all_patches = torch.cat(chunks, dim=0)
    generator = torch.Generator().manual_seed(config.seed)
    count = min(config.memory_bank_size, all_patches.shape[0])
    indices = torch.randperm(all_patches.shape[0], generator=generator)[:count]
    return F.normalize(all_patches[indices].float(), dim=-1)


def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel_1d = torch.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)[None, None]


def smooth_map(anomaly_map: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel(sigma, anomaly_map.device)
    padding = kernel.shape[-1] // 2
    padded = F.pad(anomaly_map, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(padded, kernel)


@torch.inference_mode()
def nearest_memory_distance(
    query: torch.Tensor,
    memory_bank: torch.Tensor,
    bank_chunk_size: int = 4096,
) -> torch.Tensor:
    """Cosine nearest-neighbor distance without materializing the full matrix."""
    best_similarity = torch.full(
        (query.shape[0],),
        -1.0,
        device=query.device,
        dtype=query.dtype,
    )
    for start in range(0, memory_bank.shape[0], bank_chunk_size):
        bank_chunk = memory_bank[start:start + bank_chunk_size]
        similarities = query @ bank_chunk.T
        best_similarity = torch.maximum(best_similarity, similarities.max(dim=1).values)
    return (1.0 - best_similarity).clamp_min(0.0)


@torch.inference_mode()
def score_embeddings(
    embeddings: torch.Tensor,
    memory_bank: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    maps = []
    image_scores = []
    grid = config.feature_grid_size
    for embedding in embeddings:
        distances = nearest_memory_distance(embedding, memory_bank)
        anomaly_map = distances.reshape(1, 1, grid, grid)
        anomaly_map = smooth_map(anomaly_map, config.gaussian_sigma)
        maps.append(anomaly_map)

        flat = anomaly_map.flatten()
        top_count = max(1, int(math.ceil(flat.numel() * config.image_top_fraction)))
        image_scores.append(flat.topk(top_count).values.mean())
    return torch.cat(maps, dim=0), torch.stack(image_scores)


def fit_calibration(pixel_values: np.ndarray, image_values: np.ndarray) -> dict[str, float]:
    pixel_low = float(np.quantile(pixel_values, 0.05))
    pixel_scale = max(float(np.quantile(pixel_values, 0.995)) - pixel_low, 1e-8)
    image_low = float(np.quantile(image_values, 0.05))
    image_scale = max(float(np.quantile(image_values, 0.995)) - image_low, 1e-8)
    return {
        "pixel_low": pixel_low,
        "pixel_scale": pixel_scale,
        "image_low": image_low,
        "image_scale": image_scale,
    }


def normalize_score(value: torch.Tensor, low: float, scale: float) -> torch.Tensor:
    # Exponential scaling is bounded but preserves ordering beyond the normal range.
    return 1.0 - torch.exp(-torch.clamp(value - low, min=0.0) / scale)


@torch.inference_mode()
def collect_calibration(
    extractor: nn.Module,
    loader: DataLoader,
    projection: torch.Tensor,
    memory_bank: torch.Tensor,
    config: Config,
    device: torch.device,
) -> dict[str, float]:
    pixel_samples: list[np.ndarray] = []
    image_samples: list[np.ndarray] = []
    for batch_index, (images, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        embeddings = extract_features(extractor, images, projection, config)
        maps, scores = score_embeddings(embeddings, memory_bank, config)
        pixel_samples.append(maps.cpu().numpy().ravel())
        image_samples.append(scores.cpu().numpy())
        print(f"\rCalibrating: batch {batch_index}/{len(loader)}", end="", flush=True)
    print()
    return fit_calibration(
        np.concatenate(pixel_samples),
        np.concatenate(image_samples),
    )


def save_model(
    path: Path,
    memory_bank: torch.Tensor,
    projection: torch.Tensor,
    calibration: dict[str, float],
    config: Config,
) -> None:
    payload = {
        "memory_bank": memory_bank.cpu(),
        "projection": projection.cpu(),
        "calibration": calibration,
        "backbone": "wide_resnet50_2",
        "feature_layers": config.feature_layers,
        "image_size": config.image_size,
        "feature_grid_size": config.feature_grid_size,
    }
    torch.save(payload, path)

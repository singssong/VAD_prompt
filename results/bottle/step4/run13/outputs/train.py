#!/usr/bin/env python3
"""Train a feature-memory anomaly detector using normal images only."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class Config:
    train_dir: str = "data/train"
    model_path: str = "outputs/model.pt"
    image_size: int = 256
    layers: tuple[str, ...] = ("layer2", "layer3")
    batch_size: int = 16
    calibration_fraction: float = 0.15
    patches_per_image: int = 256
    memory_size: int = 45000
    distance_chunk: int = 256
    gaussian_sigma: float = 1.5
    top_fraction: float = 0.01
    seed: int = 13


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int):
        self.paths = paths
        self.image_size = image_size

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
        mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
        std = torch.tensor(IMAGENET_STD)[:, None, None]
        return (tensor - mean) / std, path.name


class MidLevelExtractor(nn.Module):
    """ImageNet ResNet-18 with hooks on two mid-level feature stages."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(x)
        layer3 = self.backbone.layer3(layer2)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        features = torch.cat((layer2, layer3), dim=1)
        return F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


@torch.inference_mode()
def extract_features(
    extractor: nn.Module, loader: DataLoader, device: torch.device
) -> list[torch.Tensor]:
    """Extract concatenated layer2/layer3 patch grids, one tensor per image."""
    result: list[torch.Tensor] = []
    for images, _ in loader:
        grids = extractor(images.to(device)).cpu()
        result.extend(grids[i] for i in range(grids.shape[0]))
    return result


def build_normal_model(
    feature_grids: list[torch.Tensor], config: Config
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate channel scaling and store a representative normal patch bank."""
    all_patches = torch.cat(
        [grid.permute(1, 2, 0).reshape(-1, grid.shape[0]) for grid in feature_grids]
    )
    channel_mean = all_patches.mean(dim=0)
    channel_std = all_patches.std(dim=0).clamp_min(1e-6)

    generator = torch.Generator().manual_seed(config.seed)
    sampled = []
    for grid in feature_grids:
        patches = grid.permute(1, 2, 0).reshape(-1, grid.shape[0])
        count = min(config.patches_per_image, patches.shape[0])
        sampled.append(patches[torch.randperm(patches.shape[0], generator=generator)[:count]])
    memory = torch.cat(sampled)
    if memory.shape[0] > config.memory_size:
        indices = torch.randperm(memory.shape[0], generator=generator)[: config.memory_size]
        memory = memory[indices]
    memory = (memory - channel_mean) / channel_std
    memory = F.normalize(memory, dim=1)
    return memory.contiguous(), channel_mean, channel_std


def nearest_distances(
    queries: torch.Tensor, memory: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Compute cosine nearest-neighbor distance without a huge distance matrix."""
    output = []
    memory_t = memory.T.contiguous()
    for start in range(0, queries.shape[0], chunk_size):
        similarity = queries[start : start + chunk_size] @ memory_t
        output.append(1.0 - similarity.max(dim=1).values)
    return torch.cat(output)


def gaussian_smooth(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(round(3 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=maps.device, dtype=maps.dtype)
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel = kernel_1d[:, None] @ kernel_1d[None, :]
    return F.conv2d(maps, kernel[None, None], padding=radius)


def score_feature_grid(
    grid: torch.Tensor,
    memory: torch.Tensor,
    channel_mean: torch.Tensor,
    channel_std: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, float]:
    """Return a smoothed patch anomaly map and top-tail image score."""
    height, width = grid.shape[-2:]
    patches = grid.permute(1, 2, 0).reshape(-1, grid.shape[0])
    patches = F.normalize((patches - channel_mean) / channel_std, dim=1)
    anomaly_map = nearest_distances(patches, memory, config.distance_chunk)
    anomaly_map = gaussian_smooth(anomaly_map.reshape(1, 1, height, width), config.gaussian_sigma)
    flat = anomaly_map.flatten()
    top_count = max(1, int(round(flat.numel() * config.top_fraction)))
    image_score = flat.topk(top_count).values.mean().item()
    return anomaly_map[0, 0].cpu(), image_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default=Config.train_dir)
    parser.add_argument("--model-path", default=Config.model_path)
    args = parser.parse_args()
    config = Config(train_dir=args.train_dir, model_path=args.model_path)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = list_images(Path(config.train_dir))
    if len(paths) < 2:
        raise RuntimeError(f"Need at least two training images in {config.train_dir}")
    shuffled = paths.copy()
    random.Random(config.seed).shuffle(shuffled)
    calibration_count = max(1, round(len(shuffled) * config.calibration_fraction))
    calibration_paths = shuffled[:calibration_count]
    fit_paths = shuffled[calibration_count:]

    extractor = MidLevelExtractor().to(device).eval()
    fit_loader = DataLoader(
        ImageDataset(fit_paths, config.image_size),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    fit_features = extract_features(extractor, fit_loader, device)
    memory, channel_mean, channel_std = build_normal_model(fit_features, config)

    calibration_loader = DataLoader(
        ImageDataset(calibration_paths, config.image_size),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    calibration_features = extract_features(extractor, calibration_loader, device)
    memory_device = memory.to(device)
    mean_device = channel_mean.to(device)
    std_device = channel_std.to(device)
    image_scores, pixel_scores = [], []
    for grid in calibration_features:
        anomaly_map, image_score = score_feature_grid(
            grid.to(device), memory_device, mean_device, std_device, config
        )
        image_scores.append(image_score)
        pixel_scores.extend(anomaly_map.flatten().tolist())

    image_low, image_high = np.quantile(image_scores, [0.05, 0.995]).tolist()
    pixel_low, pixel_high = np.quantile(pixel_scores, [0.50, 0.999]).tolist()
    eps = 1e-8
    calibration = {
        "image_low": float(image_low),
        "image_high": float(max(image_high, image_low + eps)),
        "pixel_low": float(pixel_low),
        "pixel_high": float(max(pixel_high, pixel_low + eps)),
    }

    model_path = Path(config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "backbone": "resnet18_imagenet1k_v1",
            "memory": memory,
            "channel_mean": channel_mean,
            "channel_std": channel_std,
            "calibration": calibration,
        },
        model_path,
    )
    metadata = {
        "fit_images": len(fit_paths),
        "calibration_images": len(calibration_paths),
        "memory_patches": memory.shape[0],
        "feature_channels": memory.shape[1],
        "calibration": calibration,
    }
    model_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved normal feature model to {model_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

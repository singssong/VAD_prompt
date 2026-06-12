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
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_files(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, paths: Iterable[Path]) -> None:
        self.paths = list(paths)
        self.transform = Compose([
            Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


def make_loader(paths: Iterable[Path], shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ImageDataset(paths),
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.NUM_WORKERS > 0,
    )


class FeatureExtractor(nn.Module):
    """ImageNet ResNet-18 truncated after layer3."""

    def __init__(self) -> None:
        super().__init__()
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            network.conv1, network.bn1, network.relu, network.maxpool
        )
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
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
        # Normalize each depth independently so neither layer dominates distance.
        split = level2.shape[1]
        shallow = F.normalize(features[:, :split], dim=1)
        deep = F.normalize(features[:, split:], dim=1)
        return torch.cat((shallow, deep), dim=1)


def feature_patches(feature_map: torch.Tensor) -> torch.Tensor:
    """Convert BxCxHxW features to Bx(HW)xC patch vectors."""
    return feature_map.permute(0, 2, 3, 1).flatten(1, 2).contiguous()


@torch.inference_mode()
def build_normal_model(
    extractor: FeatureExtractor,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Extract normal patches and retain a deterministic random memory bank."""
    batches = []
    for images, _ in loader:
        features = extractor(images.to(device, non_blocking=True))
        batches.append(feature_patches(features).cpu())

    all_patches = torch.cat(batches, dim=0).flatten(0, 1)
    generator = torch.Generator().manual_seed(config.SEED)
    count = min(config.MEMORY_BANK_SIZE, all_patches.shape[0])
    indices = torch.randperm(all_patches.shape[0], generator=generator)[:count]
    return all_patches[indices].contiguous()


def nearest_patch_distances(
    query: torch.Tensor,
    memory_bank: torch.Tensor,
    memory_chunk_size: int = 3000,
) -> torch.Tensor:
    """Exact nearest-neighbor Euclidean distance, chunked to cap GPU memory."""
    query_norm = (query * query).sum(dim=1, keepdim=True)
    best_squared = torch.full(
        (query.shape[0],), float("inf"), device=query.device, dtype=query.dtype
    )
    for memory_chunk in memory_bank.split(memory_chunk_size):
        memory_norm = (memory_chunk * memory_chunk).sum(dim=1).unsqueeze(0)
        squared = query_norm + memory_norm - 2.0 * (query @ memory_chunk.T)
        best_squared = torch.minimum(best_squared, squared.min(dim=1).values)
    return best_squared.clamp_min_(0).sqrt_()


def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, math.ceil(3 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel_1d = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)[None, None]


def smooth_and_resize(
    maps: torch.Tensor, output_size: int = config.IMAGE_SIZE
) -> torch.Tensor:
    kernel = gaussian_kernel(config.GAUSSIAN_SIGMA, maps.device)
    padding = kernel.shape[-1] // 2
    maps = F.pad(maps[:, None], (padding,) * 4, mode="reflect")
    maps = F.conv2d(maps, kernel)
    return F.interpolate(
        maps, size=(output_size, output_size), mode="bilinear", align_corners=False
    )[:, 0]


def aggregate_image_scores(pixel_maps: torch.Tensor) -> torch.Tensor:
    flattened = pixel_maps.flatten(1)
    top_count = max(1, round(flattened.shape[1] * config.IMAGE_TOP_FRACTION))
    return flattened.topk(top_count, dim=1).values.mean(dim=1)


@torch.inference_mode()
def score_loader(
    extractor: FeatureExtractor,
    memory_bank_cpu: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Return filenames, raw image scores, and raw 256x256 anomaly maps."""
    memory_bank = memory_bank_cpu.to(device)
    all_names: list[str] = []
    all_image_scores = []
    all_pixel_maps = []

    for images, names in loader:
        features = extractor(images.to(device, non_blocking=True))
        patches = feature_patches(features)
        batch_size, patch_count, channels = patches.shape
        distances = nearest_patch_distances(
            patches.reshape(-1, channels), memory_bank
        )
        height, width = features.shape[-2:]
        low_resolution_maps = distances.reshape(batch_size, height, width)
        pixel_maps = smooth_and_resize(low_resolution_maps)
        image_scores = aggregate_image_scores(pixel_maps)

        all_names.extend(names)
        all_image_scores.append(image_scores.cpu())
        all_pixel_maps.append(pixel_maps.cpu())

    return (
        all_names,
        torch.cat(all_image_scores),
        torch.cat(all_pixel_maps),
    )


def robust_calibration(values: torch.Tensor) -> dict[str, float]:
    flattened = values.float().flatten()
    if flattened.numel() > 1_000_000:
        generator = torch.Generator().manual_seed(config.SEED)
        indices = torch.randint(
            flattened.numel(), (1_000_000,), generator=generator
        )
        flattened = flattened[indices]
    baseline = torch.quantile(flattened, 0.05).item()
    upper = torch.quantile(flattened, 0.995).item()
    scale = max(upper - baseline, 1e-6)
    return {"baseline": baseline, "scale": scale}


def normalize_scores(
    values: torch.Tensor, calibration: dict[str, float]
) -> torch.Tensor:
    shifted = (values.float() - calibration["baseline"]).clamp_min(0)
    return 1.0 - torch.exp(-shifted / calibration["scale"])

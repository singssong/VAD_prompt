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
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

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
        if path.is_file() and path.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.transform = Compose([
            Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
            ToTensor(),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


class FeatureExtractor(nn.Module):
    """Extract and align mid-level Wide ResNet feature maps."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self._features: dict[str, torch.Tensor] = {}
        self._hooks = [
            getattr(self.backbone, layer).register_forward_hook(self._capture(layer))
            for layer in config.FEATURE_LAYERS
        ]

    def _capture(self, name: str):
        def hook(_module, _inputs, output):
            self._features[name] = output
        return hook

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self._features.clear()
        self.backbone(images)
        aligned = []
        for layer in config.FEATURE_LAYERS:
            feature = self._features[layer]
            feature = F.avg_pool2d(
                feature,
                kernel_size=config.LOCAL_AVG_POOL_KERNEL,
                stride=1,
                padding=config.LOCAL_AVG_POOL_KERNEL // 2,
            )
            feature = F.interpolate(
                feature,
                size=(config.FEATURE_GRID_SIZE, config.FEATURE_GRID_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            aligned.append(F.normalize(feature, dim=1))
        descriptors = torch.cat(aligned, dim=1)
        return F.normalize(descriptors, dim=1)


def make_loader(paths: list[Path], batch_size: int) -> DataLoader:
    return DataLoader(
        ImageDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def extract_feature_batches(
    extractor: FeatureExtractor,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
):
    with torch.inference_mode():
        for images, names in make_loader(paths, batch_size):
            features = extractor(images.to(device, non_blocking=True))
            yield features.cpu(), list(names)


def flatten_descriptors(feature_map: torch.Tensor) -> torch.Tensor:
    return feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])


def build_memory_bank(
    extractor: FeatureExtractor,
    paths: list[Path],
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(config.SEED)
    sampled = []
    patch_count = config.FEATURE_GRID_SIZE ** 2
    take = min(config.PATCHES_PER_TRAIN_IMAGE, patch_count)
    for features, _ in extract_feature_batches(
        extractor, paths, device, config.TRAIN_BATCH_SIZE
    ):
        for feature in features:
            patches = flatten_descriptors(feature.unsqueeze(0))
            indices = torch.randperm(patch_count, generator=generator)[:take]
            sampled.append(patches[indices])
    bank = torch.cat(sampled, dim=0)
    if len(bank) > config.MAX_MEMORY_PATCHES:
        indices = torch.randperm(len(bank), generator=generator)[:config.MAX_MEMORY_PATCHES]
        bank = bank[indices]
    return bank.contiguous()


def nearest_patch_distances(
    descriptors: torch.Tensor,
    memory_bank: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    bank = memory_bank.to(device=device, dtype=torch.float32)
    flat = flatten_descriptors(descriptors).to(device=device, dtype=torch.float32)
    distances = []
    for start in range(0, len(flat), config.DISTANCE_QUERY_CHUNK):
        query = flat[start:start + config.DISTANCE_QUERY_CHUNK]
        distances.append(torch.cdist(query, bank).amin(dim=1).cpu())
    return torch.cat(distances).reshape(
        descriptors.shape[0], config.FEATURE_GRID_SIZE, config.FEATURE_GRID_SIZE
    )


def gaussian_smooth(anomaly_maps: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(3 * sigma))
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel_1d = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, *kernel_2d.shape)
    maps = anomaly_maps.unsqueeze(1)
    maps = F.pad(maps, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(maps, kernel).squeeze(1)


def resize_anomaly_maps(anomaly_maps: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        anomaly_maps.unsqueeze(1),
        size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def aggregate_image_scores(anomaly_maps: torch.Tensor) -> torch.Tensor:
    flat = anomaly_maps.flatten(1)
    count = max(1, math.ceil(flat.shape[1] * config.IMAGE_TOP_FRACTION))
    return flat.topk(count, dim=1).values.mean(dim=1)


def score_feature_maps(
    feature_maps: torch.Tensor,
    memory_bank: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    patch_maps = nearest_patch_distances(feature_maps, memory_bank, device)
    smoothed = gaussian_smooth(patch_maps, config.GAUSSIAN_SIGMA)
    image_scores = aggregate_image_scores(smoothed)
    pixel_maps = resize_anomaly_maps(smoothed)
    return pixel_maps, image_scores


def robust_scale(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    denominator = max(high - low, 1e-8)
    standardized = ((value - low) / denominator).clamp_min(0.0)
    return 1.0 - torch.exp(-standardized)

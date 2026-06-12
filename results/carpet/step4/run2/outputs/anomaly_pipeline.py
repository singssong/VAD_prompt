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
    def __init__(self, paths: Iterable[Path]):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image,
                [config.IMAGE_SIZE, config.IMAGE_SIZE],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.pil_to_tensor(image).float().div_(255.0)
        tensor = TF.normalize(
            tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return tensor, path.name


def list_images(directory: Path) -> list[Path]:
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in config.VALID_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"No supported images found in {directory}")
    return paths


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
    def __init__(self):
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        x = self.stem(images)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)

        # Local averaging stabilizes texture features before the two depths are joined.
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        features = torch.cat([layer2, layer3], dim=1)
        return F.normalize(features, dim=1)


def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_projection(input_dim: int, output_dim: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(config.SEED)
    matrix = torch.randn(input_dim, output_dim, generator=generator)
    matrix = torch.linalg.qr(matrix, mode="reduced").Q
    return matrix.to(device)


@torch.inference_mode()
def extract_features(
    extractor: nn.Module,
    loader: DataLoader,
    projection: torch.Tensor | None,
    device: torch.device,
):
    for images, names in loader:
        feature_map = extractor(images.to(device, non_blocking=True))
        batch, channels, height, width = feature_map.shape
        patches = feature_map.permute(0, 2, 3, 1).reshape(-1, channels)
        if projection is not None:
            patches = patches @ projection
            patches = F.normalize(patches, dim=1)
        yield patches.reshape(batch, height, width, -1), list(names)


def build_normal_feature_model(
    extractor: nn.Module,
    train_paths: list[Path],
    device: torch.device,
):
    calibration_count = max(1, round(len(train_paths) * config.CALIBRATION_FRACTION))
    generator = torch.Generator().manual_seed(config.SEED)
    order = torch.randperm(len(train_paths), generator=generator).tolist()
    calibration_indices = set(order[:calibration_count])
    memory_paths = [
        path for index, path in enumerate(train_paths) if index not in calibration_indices
    ]
    calibration_paths = [
        path for index, path in enumerate(train_paths) if index in calibration_indices
    ]

    probe = next(iter(make_loader(memory_paths[:1])))[0].to(device)
    with torch.inference_mode():
        input_dim = extractor(probe).shape[1]
    projection = create_projection(input_dim, config.PROJECTION_DIM, device)

    all_patches = []
    for feature_maps, _ in extract_features(
        extractor, make_loader(memory_paths), projection, device
    ):
        all_patches.append(feature_maps.reshape(-1, config.PROJECTION_DIM).cpu())
    all_patches = torch.cat(all_patches)

    sample_generator = torch.Generator().manual_seed(config.SEED)
    selected = torch.randperm(all_patches.shape[0], generator=sample_generator)[
        : config.MAX_MEMORY_PATCHES
    ]
    memory_bank = all_patches[selected].contiguous()
    return memory_bank, projection.cpu(), calibration_paths


def nearest_neighbor_distances(
    queries: torch.Tensor,
    memory_bank: torch.Tensor,
) -> torch.Tensor:
    output = []
    for query_start in range(0, len(queries), config.DISTANCE_QUERY_CHUNK):
        query = queries[query_start : query_start + config.DISTANCE_QUERY_CHUNK]
        best = torch.full(
            (len(query),), float("inf"), dtype=query.dtype, device=query.device
        )
        for bank_start in range(0, len(memory_bank), config.DISTANCE_BANK_CHUNK):
            bank = memory_bank[
                bank_start : bank_start + config.DISTANCE_BANK_CHUNK
            ]
            distances = torch.cdist(query, bank)
            best = torch.minimum(best, distances.min(dim=1).values)
        output.append(best)
    return torch.cat(output)


def gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype):
    radius = max(1, math.ceil(3 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    size = kernel_1d.numel()
    return torch.outer(kernel_1d, kernel_1d).view(1, 1, size, size)


def smooth_and_resize_maps(patch_maps: torch.Tensor) -> torch.Tensor:
    maps = patch_maps.unsqueeze(1)
    kernel = gaussian_kernel(config.GAUSSIAN_SIGMA, maps.device, maps.dtype)
    padding = kernel.shape[-1] // 2
    maps = F.pad(maps, (padding,) * 4, mode="reflect")
    maps = F.conv2d(maps, kernel)
    maps = F.interpolate(
        maps,
        size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    return maps.squeeze(1)


def aggregate_image_scores(pixel_maps: torch.Tensor) -> torch.Tensor:
    flat = pixel_maps.flatten(1)
    top_count = max(1, round(flat.shape[1] * config.IMAGE_TOP_FRACTION))
    return flat.topk(top_count, dim=1).values.mean(dim=1)


@torch.inference_mode()
def score_images(
    extractor: nn.Module,
    paths: list[Path],
    memory_bank: torch.Tensor,
    projection: torch.Tensor,
    device: torch.device,
):
    bank = memory_bank.to(device)
    projection = projection.to(device)
    for feature_maps, names in extract_features(
        extractor, make_loader(paths), projection, device
    ):
        batch, height, width, channels = feature_maps.shape
        distances = nearest_neighbor_distances(
            feature_maps.reshape(-1, channels), bank
        )
        patch_maps = distances.reshape(batch, height, width)
        pixel_maps = smooth_and_resize_maps(patch_maps)
        image_scores = aggregate_image_scores(pixel_maps)
        yield names, pixel_maps.cpu(), image_scores.cpu()


def estimate_calibration(
    extractor: nn.Module,
    calibration_paths: list[Path],
    memory_bank: torch.Tensor,
    projection: torch.Tensor,
    device: torch.device,
):
    all_pixels = []
    all_images = []
    for _, pixel_maps, image_scores in score_images(
        extractor, calibration_paths, memory_bank, projection, device
    ):
        all_pixels.append(pixel_maps.flatten())
        all_images.append(image_scores)
    pixels = torch.cat(all_pixels)
    images = torch.cat(all_images)
    image_low, image_high = torch.quantile(
        images, torch.tensor(config.IMAGE_CALIBRATION_QUANTILES)
    ).tolist()
    pixel_low, pixel_high = torch.quantile(
        pixels, torch.tensor(config.PIXEL_CALIBRATION_QUANTILES)
    ).tolist()
    return {
        "image_low": float(image_low),
        "image_high": float(max(image_high, image_low + 1e-6)),
        "pixel_low": float(pixel_low),
        "pixel_high": float(max(pixel_high, pixel_low + 1e-6)),
    }


def normalize_score(value, low: float, high: float):
    if isinstance(value, torch.Tensor):
        return ((value - low) / (high - low)).clamp(0.0, 1.0)
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def normalize_image_score(value: float, low: float, high: float) -> float:
    # A robust sigmoid preserves ordering beyond the normal calibration range.
    midpoint = 0.5 * (low + high)
    scaled = 4.0 * (value - midpoint) / (high - low)
    return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, scaled)))))

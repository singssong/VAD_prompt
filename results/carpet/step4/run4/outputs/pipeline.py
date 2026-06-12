import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

import config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.transform = v2.Compose(
            [
                v2.Resize(
                    (config.IMAGE_SIZE, config.IMAGE_SIZE),
                    interpolation=InterpolationMode.BILINEAR,
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


def make_loader(paths: list[Path], shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ImageDataset(paths),
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.NUM_WORKERS > 0,
    )


class MidLevelResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
        )
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = self.layer1(features)
        level2 = self.layer2(features)
        level3 = self.layer3(level2)
        level3 = F.interpolate(
            level3,
            size=level2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # Per-location channel normalization balances the two depth levels.
        level2 = F.normalize(level2, dim=1)
        level3 = F.normalize(level3, dim=1)
        return torch.cat((level2, level3), dim=1)


def extract_features(
    backbone: nn.Module, images: torch.Tensor, device: torch.device
) -> torch.Tensor:
    with torch.inference_mode():
        return backbone(images.to(device, non_blocking=True))


def fit_normal_feature_model(
    backbone: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, torch.Tensor]:
    feature_sum = None
    feature_square_sum = None
    sample_count = 0

    for images, _ in loader:
        features = extract_features(backbone, images, device)
        descriptors = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        batch_sum = descriptors.sum(dim=0, dtype=torch.float64)
        batch_square_sum = descriptors.square().sum(dim=0, dtype=torch.float64)
        feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
        feature_square_sum = (
            batch_square_sum
            if feature_square_sum is None
            else feature_square_sum + batch_square_sum
        )
        sample_count += descriptors.shape[0]

    mean = feature_sum / sample_count
    variance = feature_square_sum / sample_count - mean.square()
    variance = variance.clamp_min(config.VARIANCE_FLOOR)
    return {
        "mean": mean.float().cpu(),
        "variance": variance.float().cpu(),
        "sample_count": torch.tensor(sample_count),
    }


def gaussian_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    size = config.GAUSSIAN_KERNEL_SIZE
    radius = size // 2
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * config.GAUSSIAN_SIGMA**2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, size, size)


def score_feature_maps(
    features: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    mean = mean.to(features.device).view(1, -1, 1, 1)
    variance = variance.to(features.device).view(1, -1, 1, 1)
    anomaly_map = torch.sqrt(((features - mean).square() / variance).mean(dim=1))
    kernel = gaussian_kernel(anomaly_map.device, anomaly_map.dtype)
    anomaly_map = F.conv2d(
        anomaly_map.unsqueeze(1),
        kernel,
        padding=config.GAUSSIAN_KERNEL_SIZE // 2,
    )
    return anomaly_map.squeeze(1)


def resize_maps(anomaly_maps: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        anomaly_maps.unsqueeze(1),
        size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def aggregate_image_scores(anomaly_maps: torch.Tensor) -> torch.Tensor:
    flat_maps = anomaly_maps.flatten(1)
    top_count = max(1, math.ceil(flat_maps.shape[1] * config.TOP_FRACTION))
    return flat_maps.topk(top_count, dim=1).values.mean(dim=1)


def normalize_scores(
    values: torch.Tensor, low: float | torch.Tensor, high: float | torch.Tensor
) -> torch.Tensor:
    high_value = torch.as_tensor(high, dtype=values.dtype, device=values.device)
    # A rational mapping stays in [0, 1) without clipping anomalous-score ranks.
    scaled = values.clamp_min(0.0) / high_value.clamp_min(1.0e-8)
    return scaled / (1.0 + scaled)


def collect_training_calibration(
    backbone: nn.Module,
    loader: DataLoader,
    model: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    image_scores = []
    pixel_samples = []
    for images, _ in loader:
        features = extract_features(backbone, images, device)
        low_resolution_maps = score_feature_maps(
            features, model["mean"], model["variance"]
        )
        image_scores.append(aggregate_image_scores(low_resolution_maps).cpu())
        pixel_samples.append(low_resolution_maps.flatten().cpu())

    image_scores = torch.cat(image_scores)
    pixel_scores = torch.cat(pixel_samples)
    return {
        "image_low": torch.quantile(
            image_scores, config.IMAGE_SCORE_LOW_QUANTILE
        ),
        "image_high": torch.quantile(
            image_scores, config.IMAGE_SCORE_HIGH_QUANTILE
        ),
        "pixel_low": torch.quantile(
            pixel_scores, config.PIXEL_SCORE_LOW_QUANTILE
        ),
        "pixel_high": torch.quantile(
            pixel_scores, config.PIXEL_SCORE_HIGH_QUANTILE
        ),
    }

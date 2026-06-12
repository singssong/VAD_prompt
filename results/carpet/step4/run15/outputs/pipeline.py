from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

import config


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, directory: Path):
        self.files = image_files(directory)
        if not self.files:
            raise RuntimeError(f"No supported images found in {directory}")
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms(
            crop_size=config.IMAGE_SIZE,
            resize_size=config.IMAGE_SIZE,
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.files[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            tensor = self.transform(image)
        return tensor, path.name


class MidLevelFeatureExtractor(nn.Module):
    """Extract and concatenate aligned ResNet layer2/layer3 patch descriptors."""

    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = self.layer1(x)
        level2 = self.layer2(x)
        level3 = self.layer3(level2)
        level2 = F.adaptive_avg_pool2d(
            level2, (config.PATCH_GRID_SIZE, config.PATCH_GRID_SIZE)
        )
        level3 = F.adaptive_avg_pool2d(
            level3, (config.PATCH_GRID_SIZE, config.PATCH_GRID_SIZE)
        )
        features = torch.cat([level2, level3], dim=1)
        return features.permute(0, 2, 3, 1).contiguous()


def make_projection(input_dim: int, output_dim: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(config.SEED)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    projection /= np.sqrt(output_dim)
    return projection


@torch.inference_mode()
def extract_features(
    extractor: nn.Module,
    images: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Return L2-normalized projected patch features shaped B x H x W x D."""
    features = extractor(images)
    features = features @ projection
    return F.normalize(features, p=2, dim=-1)


def build_normal_feature_model(
    feature_batches: Iterable[torch.Tensor],
    bank_size: int,
) -> torch.Tensor:
    """Store a deterministic random subset of normal patch descriptors."""
    all_features = torch.cat(
        [batch.reshape(-1, batch.shape[-1]).cpu() for batch in feature_batches],
        dim=0,
    )
    generator = torch.Generator().manual_seed(config.SEED)
    if len(all_features) > bank_size:
        indices = torch.randperm(len(all_features), generator=generator)[:bank_size]
        all_features = all_features[indices]
    return F.normalize(all_features, p=2, dim=1).contiguous()


@torch.inference_mode()
def nearest_neighbor_distances(
    queries: torch.Tensor,
    memory_bank: torch.Tensor,
) -> torch.Tensor:
    """Cosine distance to the closest normal patch, computed in bounded chunks."""
    flat = queries.reshape(-1, queries.shape[-1])
    distances = []
    bank_t = memory_bank.T.contiguous()
    for start in range(0, len(flat), config.NN_QUERY_CHUNK):
        similarity = flat[start:start + config.NN_QUERY_CHUNK] @ bank_t
        distances.append(1.0 - similarity.max(dim=1).values)
    return torch.cat(distances).reshape(queries.shape[:-1])


def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel_1d = torch.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d[None, None]


def score_feature_maps(
    features: torch.Tensor,
    memory_bank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create smoothed pixel maps and top-tail image anomaly scores."""
    patch_maps = nearest_neighbor_distances(features, memory_bank).unsqueeze(1)
    kernel = gaussian_kernel(config.GAUSSIAN_SIGMA, patch_maps.device)
    padding = kernel.shape[-1] // 2
    patch_maps = F.pad(patch_maps, (padding,) * 4, mode="reflect")
    patch_maps = F.conv2d(patch_maps, kernel)
    pixel_maps = F.interpolate(
        patch_maps,
        size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    top_count = max(
        1,
        int(config.IMAGE_SIZE * config.IMAGE_SIZE * config.IMAGE_TOP_FRACTION),
    )
    image_scores = pixel_maps.flatten(1).topk(top_count, dim=1).values.mean(dim=1)
    return pixel_maps, image_scores


def robust_normalize(
    values: np.ndarray,
    low_percentile: float = config.NORMALIZATION_LOW_PERCENTILE,
    high_percentile: float = config.NORMALIZATION_HIGH_PERCENTILE,
) -> np.ndarray:
    low, high = np.percentile(values, [low_percentile, high_percentile])
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def make_loader(directory: Path, batch_size: int) -> DataLoader:
    return DataLoader(
        ImageDataset(directory),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import functional as TF

import config


class ImageDataset(Dataset):
    def __init__(self, paths: Iterable[Path]) -> None:
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            tensor = TF.pil_to_tensor(image).float().div_(255.0)
        tensor = TF.normalize(
            tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return tensor, path.name


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )


class FeatureExtractor(nn.Module):
    """Extract and align mid-level Wide ResNet features."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: Tensor) -> Tensor:
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(x)
        layer3 = self.backbone.layer3(layer2)

        layer2 = F.normalize(layer2, dim=1)
        layer3 = F.normalize(layer3, dim=1)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        features = torch.cat((layer2, layer3), dim=1)
        return F.avg_pool2d(
            features,
            kernel_size=config.PATCH_POOL_KERNEL,
            stride=1,
            padding=config.PATCH_POOL_KERNEL // 2,
        )


def make_loader(paths: list[Path], shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ImageDataset(paths),
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def extract_features(
    extractor: FeatureExtractor, images: Tensor, device: torch.device
) -> Tensor:
    return extractor(images.to(device, non_blocking=True))


@torch.inference_mode()
def build_normal_feature_model(
    extractor: FeatureExtractor,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    feature_chunks = []
    spatial_shape = None
    for images, _ in loader:
        features = extract_features(extractor, images, device)
        spatial_shape = tuple(features.shape[-2:])
        descriptors = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        feature_chunks.append(descriptors.cpu())

    all_features = torch.cat(feature_chunks, dim=0)
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    if len(all_features) > config.MEMORY_BANK_SIZE:
        indices = torch.randperm(len(all_features), generator=generator)[
            : config.MEMORY_BANK_SIZE
        ]
        memory_bank = all_features[indices]
    else:
        memory_bank = all_features

    return {
        "memory_bank": memory_bank.contiguous(),
        "spatial_shape": spatial_shape,
        "backbone": config.BACKBONE,
        "feature_layers": config.FEATURE_LAYERS,
        "image_size": config.IMAGE_SIZE,
    }


def nearest_neighbor_distances(queries: Tensor, memory_bank: Tensor) -> Tensor:
    distances = []
    for query_batch in queries.split(config.DISTANCE_QUERY_BATCH):
        distances.append(torch.cdist(query_batch, memory_bank).amin(dim=1))
    return torch.cat(distances)


@torch.inference_mode()
def score_images(
    extractor: FeatureExtractor,
    images: Tensor,
    memory_bank: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    features = extract_features(extractor, images, device)
    batch_size, channels, height, width = features.shape
    descriptors = features.permute(0, 2, 3, 1).reshape(-1, channels)
    patch_scores = nearest_neighbor_distances(descriptors, memory_bank)
    maps = patch_scores.reshape(batch_size, 1, height, width)

    maps = TF.gaussian_blur(
        maps,
        kernel_size=[config.GAUSSIAN_KERNEL, config.GAUSSIAN_KERNEL],
        sigma=[config.GAUSSIAN_SIGMA, config.GAUSSIAN_SIGMA],
    )
    maps = F.interpolate(
        maps,
        size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    flat_maps = maps.flatten(1)
    top_count = max(1, round(flat_maps.shape[1] * config.IMAGE_SCORE_TOP_FRACTION))
    image_scores = flat_maps.topk(top_count, dim=1).values.mean(dim=1)
    return maps, image_scores


def minmax_normalize(values: Tensor) -> Tensor:
    minimum = values.min()
    span = values.max() - minimum
    if span <= 1e-12:
        return torch.zeros_like(values)
    return (values - minimum) / span

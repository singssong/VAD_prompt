import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import config


class ImageDataset(Dataset):
    def __init__(self, directory: Path):
        self.paths = sorted(path for path in directory.iterdir() if path.is_file())
        if not self.paths:
            raise RuntimeError(f"No images found in {directory}")
        self.transform = Compose(
            [
                Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
                ToTensor(),
                Normalize(
                    mean=ResNet18_Weights.DEFAULT.transforms().mean,
                    std=ResNet18_Weights.DEFAULT.transforms().std,
                ),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


class MidLevelResNet18(nn.Module):
    """ImageNet ResNet-18 exposing aligned features from three depths."""

    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        x = self.stem(images)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        features = (layer1, layer2, layer3)
        aligned = [
            F.interpolate(
                feature,
                size=(config.FEATURE_SIZE, config.FEATURE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            for feature in features
        ]
        return torch.cat(aligned, dim=1)


def extract_features(backbone, images):
    """Extract concatenated, spatially aligned mid-level CNN features."""
    with torch.inference_mode():
        return backbone(images)


def gaussian_smooth(maps, sigma=config.GAUSSIAN_SIGMA):
    radius = max(1, math.ceil(3 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=maps.device)
    kernel = torch.exp(-(coordinates.float() ** 2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel_x = kernel.view(1, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1)
    maps = F.pad(maps, (radius, radius, 0, 0), mode="reflect")
    maps = F.conv2d(maps, kernel_x)
    maps = F.pad(maps, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(maps, kernel_y)


def score_feature_maps(features, model):
    """Return one Mahalanobis-like anomaly value per feature-map position."""
    mean = model["mean"].to(features.device)
    variance = model["variance"].to(features.device)
    squared_distance = (features - mean).square() / variance
    return torch.sqrt(squared_distance.mean(dim=1, keepdim=True) + 1e-12)


def postprocess_maps(raw_maps):
    smoothed = gaussian_smooth(raw_maps)
    return F.interpolate(
        smoothed,
        size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )


def aggregate_image_scores(pixel_maps):
    flat = pixel_maps.flatten(start_dim=1)
    count = max(1, round(flat.shape[1] * config.TOP_FRACTION))
    return flat.topk(count, dim=1).values.mean(dim=1)


def bounded_normalize(values, scale):
    """Strictly monotonic train-calibrated normalization into [0, 1)."""
    scale = max(float(scale), 1e-12)
    return values / (values + scale)


def save_pixel_map(pixel_map, destination):
    array = pixel_map.detach().cpu().clamp(0, 1).mul(255).round().byte().numpy()
    Image.fromarray(array, mode="L").save(destination, format="PNG")


def quantile(values, q):
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


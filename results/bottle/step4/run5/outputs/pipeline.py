import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import config


def list_images(directory: Path):
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, directory: Path):
        self.paths = list_images(directory)
        self.transform = Compose(
            [
                Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                ToTensor(),
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
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
    """ImageNet ResNet-18 exposing three intermediate feature levels."""

    def __init__(self):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        x = self.stem(images)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        target_size = (config.FEATURE_SIZE, config.FEATURE_SIZE)
        levels = [
            F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            if f.shape[-2:] != target_size
            else f
            for f in (f1, f2, f3)
        ]
        # Normalize each depth independently so high-channel layers do not dominate.
        levels = [F.normalize(f, p=2, dim=1) for f in levels]
        return torch.cat(levels, dim=1)


@torch.inference_mode()
def extract_features(model, images):
    return model(images)


def create_normal_model(feature_batches):
    """Fit a position-aware diagonal Gaussian to normal feature tensors."""
    total = None
    total_sq = None
    count = 0
    for features in feature_batches:
        features = features.double()
        batch_sum = features.sum(dim=0)
        batch_sum_sq = features.square().sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        total_sq = batch_sum_sq if total_sq is None else total_sq + batch_sum_sq
        count += features.shape[0]

    if count < 2:
        raise ValueError("At least two normal training images are required")
    mean = total / count
    variance = (total_sq - total.square() / count) / (count - 1)
    variance = variance.clamp_min(config.VARIANCE_EPS)
    return mean.float(), variance.float(), count


def score_feature_map(features, mean, variance):
    """Return a per-location diagonal Mahalanobis anomaly map."""
    squared_z = (features - mean.unsqueeze(0)).square() / variance.unsqueeze(0)
    return squared_z.mean(dim=1).sqrt()


def gaussian_kernel(sigma, device, dtype):
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, *kernel_2d.shape), radius


def smooth_maps(maps):
    kernel, radius = gaussian_kernel(
        config.GAUSSIAN_SIGMA, maps.device, maps.dtype
    )
    return F.conv2d(maps.unsqueeze(1), kernel, padding=radius).squeeze(1)


def resize_maps(maps):
    return F.interpolate(
        maps.unsqueeze(1),
        size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def aggregate_image_scores(maps):
    flat = maps.flatten(1)
    k = max(1, int(round(flat.shape[1] * config.IMAGE_TOP_FRACTION)))
    return flat.topk(k, dim=1).values.mean(dim=1)


def normalize_scores(values, low, high):
    scale = max(float(high) - float(low), 1e-12)
    standardized = (values - float(low)) / scale
    # Strictly monotonic robust squashing: low -> 0.5, high -> 0.75.
    # Unlike clipping, this preserves ranking for scores beyond the train range.
    return 0.5 + 0.5 * standardized / (1.0 + standardized.abs())

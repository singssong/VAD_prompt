import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class ImageDataset(Dataset):
    def __init__(self, directory: Path, image_size: int, extensions: set[str]):
        self.paths = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
        if not self.paths:
            raise RuntimeError(f"No supported images found in {directory}")
        self.image_size = image_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.to_tensor(image)
        return TF.normalize(tensor, self.mean, self.std), path.name


class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        network = resnet18(weights=weights)
        self.stem = torch.nn.Sequential(
            network.conv1, network.bn1, network.relu, network.maxpool
        )
        self.layer1 = network.layer1
        self.layer2 = network.layer2

    def forward(self, images):
        x = self.stem(images)
        level1 = self.layer1(x)
        level2 = self.layer2(level1)
        return level1, level2


def extract_features(model, images, feature_size):
    """Extract and concatenate two mid-level ImageNet feature maps."""
    level1, level2 = model(images)
    level1 = F.interpolate(
        level1, size=(feature_size, feature_size), mode="bilinear", align_corners=False
    )
    level2 = F.interpolate(
        level2, size=(feature_size, feature_size), mode="bilinear", align_corners=False
    )
    features = torch.cat([level1, level2], dim=1)
    return F.normalize(features, p=2, dim=1)


@torch.no_grad()
def fit_normal_model(model, loader, device, config):
    """Fit an independent diagonal Gaussian at every feature-map location."""
    total = None
    total_sq = None
    count = 0
    for images, _ in loader:
        features = extract_features(
            model, images.to(device, non_blocking=True), config["feature_size"]
        ).double()
        batch_sum = features.sum(dim=0)
        batch_sum_sq = features.square().sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        total_sq = batch_sum_sq if total_sq is None else total_sq + batch_sum_sq
        count += features.shape[0]

    mean = total / count
    variance = total_sq / count - mean.square()
    variance = variance.clamp_min(config["variance_floor"])
    return mean.float(), variance.float(), count


def gaussian_kernel(sigma, device, dtype):
    radius = max(1, math.ceil(3 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d[None, None], radius


def smooth_maps(maps, sigma):
    kernel, radius = gaussian_kernel(sigma, maps.device, maps.dtype)
    padded = F.pad(maps[:, None], (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel).squeeze(1)


def score_features(features, mean, variance, sigma):
    """Return smoothed pixel maps and top-tail image anomaly scores."""
    squared_distance = (features - mean[None]).square() / variance[None]
    maps = torch.sqrt(squared_distance.mean(dim=1).clamp_min(0))
    return smooth_maps(maps, sigma)


def aggregate_image_scores(maps, top_fraction):
    flattened = maps.flatten(1)
    top_count = max(1, math.ceil(flattened.shape[1] * top_fraction))
    return flattened.topk(top_count, dim=1).values.mean(dim=1)


def normalize_scores(values, low, high):
    scale = max(float(high) - float(low), 1e-12)
    shifted = (values - float(low)).clamp_min(0.0)
    return shifted / (shifted + scale)

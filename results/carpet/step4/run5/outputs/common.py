from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from config import IMAGE_SIZE, VALID_EXTENSIONS


class ImageDataset(Dataset):
    def __init__(self, directory: Path):
        self.paths = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No supported images found in {directory}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image,
                [IMAGE_SIZE, IMAGE_SIZE],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.to_tensor(image)
        tensor = TF.normalize(
            tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return tensor, path.name


class MidLevelResNet18(nn.Module):
    """ImageNet ResNet-18 returning aligned layer2 and layer3 patch features."""

    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        features = self.stem(images)
        features = self.layer1(features)
        layer2 = self.layer2(features)
        layer3 = self.layer3(layer2)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat((layer2, layer3), dim=1)


def extract_features(model, images):
    """Return patch embeddings as [batch, height, width, channels]."""
    with torch.inference_mode():
        features = model(images)
    return features.permute(0, 2, 3, 1).contiguous()


def normalize_features(features, mean, std):
    features = (features - mean) / std
    return F.normalize(features, dim=-1)


def gaussian_smooth(maps, sigma):
    radius = max(1, int(round(3 * sigma)))
    coordinates = torch.arange(
        -radius, radius + 1, device=maps.device, dtype=maps.dtype
    )
    kernel = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    maps = F.conv2d(
        maps, kernel.view(1, 1, 1, -1), padding=(0, radius)
    )
    return F.conv2d(
        maps, kernel.view(1, 1, -1, 1), padding=(radius, 0)
    )


def robust_unit_scale(values, low_percentile=1.0, high_percentile=99.0):
    values = np.asarray(values, dtype=np.float32)
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    if high <= low:
        high = low + 1e-8
    return np.clip((values - low) / (high - low), 0.0, 1.0), low, high

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import config


class ImageDataset(Dataset):
    def __init__(self, files):
        self.files = list(files)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image,
                [config.IMAGE_SIZE, config.IMAGE_SIZE],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.to_tensor(image)
            tensor = TF.normalize(tensor, self.mean, self.std)
        return tensor, path.name


def list_images(root: Path):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in config.IMAGE_EXTENSIONS
        and ".ipynb_checkpoints" not in path.parts
    )


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        stem = self.stem(images)
        level1 = self.layer1(stem)
        level2 = self.layer2(level1)
        level2 = F.interpolate(
            level2, size=level1.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat((level1, level2), dim=1)


def choose_channels(total_channels):
    if config.SELECTED_CHANNELS >= total_channels:
        return torch.arange(total_channels)
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    return torch.randperm(total_channels, generator=generator)[: config.SELECTED_CHANNELS]


@torch.inference_mode()
def extract_features(extractor, images, channel_indices):
    features = extractor(images)
    return features[:, channel_indices.to(features.device)]


class OnlineSpatialGaussian:
    def __init__(self):
        self.count = 0
        self.sum = None
        self.sum_squares = None

    def update(self, features):
        values = features.detach().double().cpu()
        batch_sum = values.sum(dim=0)
        batch_sum_squares = values.square().sum(dim=0)
        if self.sum is None:
            self.sum = batch_sum
            self.sum_squares = batch_sum_squares
        else:
            self.sum += batch_sum
            self.sum_squares += batch_sum_squares
        self.count += values.shape[0]

    def finalize(self):
        mean = self.sum / self.count
        variance = self.sum_squares / self.count - mean.square()
        variance = variance.clamp_min(config.VARIANCE_EPSILON)
        return mean.float(), variance.float()


def build_gaussian_kernel(device, dtype):
    size = config.GAUSSIAN_KERNEL_SIZE
    radius = size // 2
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-coordinates.square() / (2.0 * config.GAUSSIAN_SIGMA**2))
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d).view(1, 1, size, size)


def score_feature_maps(features, mean, variance):
    squared_z = (features - mean.unsqueeze(0)).square() / variance.unsqueeze(0)
    return torch.sqrt(squared_z.mean(dim=1).clamp_min(0.0))


def postprocess_maps(maps):
    maps = maps.unsqueeze(1)
    kernel = build_gaussian_kernel(maps.device, maps.dtype)
    padding = config.GAUSSIAN_KERNEL_SIZE // 2
    maps = F.pad(maps, (padding, padding, padding, padding), mode="reflect")
    maps = F.conv2d(maps, kernel)
    maps = F.interpolate(
        maps,
        size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    return maps[:, 0]


def aggregate_image_scores(maps):
    flat = maps.flatten(1)
    top_count = max(1, math.ceil(flat.shape[1] * config.TOP_FRACTION))
    return flat.topk(top_count, dim=1).values.mean(dim=1)


def normalize_scores(values, low, high):
    scale = max(float(high) - float(low), 1.0e-6)
    normalized = np.maximum(np.asarray(values, dtype=np.float64) - float(low), 0.0)
    ratio = normalized / scale
    return ratio / (1.0 + ratio)


def save_pixel_map(path, anomaly_map, low, high):
    normalized = normalize_scores(anomaly_map, low, high)
    pixels = np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)

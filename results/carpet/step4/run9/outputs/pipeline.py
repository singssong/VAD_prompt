import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from config import (
    BATCH_SIZE,
    FEATURE_SIZE,
    GAUSSIAN_SIGMA,
    IMAGE_SIZE,
    NUM_WORKERS,
    SEED,
    SELECTED_FEATURES,
    SUPPORTED_SUFFIXES,
    TOP_FRACTION,
)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_images(directory):
    return sorted(
        path for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.transform = Compose([
            Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.Resampling.BILINEAR),
            ToTensor(),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.paths[index].name


def make_loader(paths, shuffle=False):
    return DataLoader(
        ImageDataset(paths),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


class MidLevelFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            network.conv1, network.bn1, network.relu, network.maxpool
        )
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3

    def forward(self, images):
        x = self.stem(images)
        level1 = self.layer1(x)
        level2 = self.layer2(level1)
        level3 = self.layer3(level2)
        target_size = (FEATURE_SIZE, FEATURE_SIZE)
        features = [
            F.interpolate(level, target_size, mode="bilinear", align_corners=False)
            if level.shape[-2:] != target_size else level
            for level in (level1, level2, level3)
        ]
        return torch.cat(features, dim=1)


def build_feature_extractor(device):
    extractor = MidLevelFeatureExtractor().eval().to(device)
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    return extractor


def choose_feature_channels(total_channels):
    generator = torch.Generator().manual_seed(SEED)
    return torch.randperm(total_channels, generator=generator)[:SELECTED_FEATURES]


@torch.inference_mode()
def extract_features(extractor, images, channel_indices):
    features = extractor(images)
    return features[:, channel_indices.to(features.device)]


def gaussian_kernel(sigma, device):
    radius = max(1, math.ceil(3 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    kernel_2d = torch.outer(kernel, kernel)
    return kernel_2d[None, None]


def smooth_maps(maps):
    kernel = gaussian_kernel(GAUSSIAN_SIGMA, maps.device)
    padding = kernel.shape[-1] // 2
    return F.conv2d(maps[:, None], kernel, padding=padding)[:, 0]


def raw_anomaly_maps(features, model):
    mean = model["mean"].to(features.device)
    variance = model["variance"].to(features.device)
    standardized = (features - mean) ** 2 / variance
    return smooth_maps(torch.sqrt(standardized.mean(dim=1) + 1e-12))


def aggregate_image_scores(maps):
    flattened = maps.flatten(1)
    count = max(1, int(flattened.shape[1] * TOP_FRACTION))
    return flattened.topk(count, dim=1).values.mean(dim=1)


@torch.inference_mode()
def fit_normal_model(extractor, loader, device):
    channel_indices = None
    feature_sum = None
    feature_square_sum = None
    sample_count = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        all_features = extractor(images)
        if channel_indices is None:
            channel_indices = choose_feature_channels(all_features.shape[1])
        features = all_features[:, channel_indices.to(device)]
        batch_sum = features.sum(dim=0).cpu()
        batch_square_sum = (features ** 2).sum(dim=0).cpu()
        feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
        feature_square_sum = (
            batch_square_sum
            if feature_square_sum is None
            else feature_square_sum + batch_square_sum
        )
        sample_count += features.shape[0]

    mean = feature_sum / sample_count
    variance = feature_square_sum / sample_count - mean ** 2
    variance = variance.clamp_min(1e-4)
    return {
        "channel_indices": channel_indices,
        "mean": mean,
        "variance": variance,
    }


@torch.inference_mode()
def calibrate_normal_scores(extractor, loader, model, device):
    image_scores = []
    pixel_samples = []
    indices = model["channel_indices"]

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        features = extract_features(extractor, images, indices)
        maps = raw_anomaly_maps(features, model)
        image_scores.append(aggregate_image_scores(maps).cpu())
        pixel_samples.append(maps.cpu().flatten())

    image_scores = torch.cat(image_scores).numpy()
    pixels = torch.cat(pixel_samples).numpy()
    image_center = float(np.quantile(image_scores, 0.99))
    image_iqr = float(np.quantile(image_scores, 0.75) - np.quantile(image_scores, 0.25))
    model["image_center"] = image_center
    model["image_scale"] = max(image_iqr, 1e-6)
    model["pixel_low"] = float(np.quantile(pixels, 0.50))
    model["pixel_high"] = float(np.quantile(pixels, 0.999))
    return model


def normalize_image_scores(raw_scores, model):
    z = (raw_scores - model["image_center"]) / model["image_scale"]
    return torch.sigmoid(z).clamp(0.0, 1.0)


def normalize_pixel_maps(maps, model):
    denominator = max(model["pixel_high"] - model["pixel_low"], 1e-6)
    return ((maps - model["pixel_low"]) / denominator).clamp(0.0, 1.0)


@torch.inference_mode()
def score_batch(extractor, images, model):
    features = extract_features(extractor, images, model["channel_indices"])
    maps = raw_anomaly_maps(features, model)
    raw_scores = aggregate_image_scores(maps)
    image_scores = normalize_image_scores(raw_scores, model)
    pixel_maps = normalize_pixel_maps(maps, model)
    pixel_maps = F.interpolate(
        pixel_maps[:, None],
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return image_scores, pixel_maps

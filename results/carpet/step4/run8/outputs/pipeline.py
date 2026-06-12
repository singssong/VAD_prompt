import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

import config


class ImageDataset(Dataset):
    def __init__(self, directory: Path):
        self.paths = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
        )
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms(
            crop_size=config.IMAGE_SIZE,
            resize_size=config.IMAGE_SIZE,
            antialias=True,
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            tensor = self.transform(image)
        return tensor, path.name


class MidLevelResNet18(nn.Module):
    """ImageNet backbone returning aligned layer2 and layer3 features."""

    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool, model.layer1
        )
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, x):
        x = self.stem(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        # Per-location normalization prevents one depth from dominating by scale.
        layer2 = F.normalize(layer2, p=2, dim=1)
        layer3 = F.normalize(layer3, p=2, dim=1)
        return torch.cat((layer2, layer3), dim=1)


def make_loader(directory: Path, shuffle=False):
    dataset = ImageDataset(directory)
    if not dataset.paths:
        raise RuntimeError(f"No supported images found in {directory}")
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def build_feature_extractor(device):
    model = MidLevelResNet18().to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def extract_features(model, images, device):
    return model(images.to(device, non_blocking=True))


@torch.inference_mode()
def fit_normal_model(model, loader, device):
    feature_sum = None
    feature_sq_sum = None
    count = 0
    for images, _ in loader:
        features = extract_features(model, images, device).double()
        batch_sum = features.sum(dim=0)
        batch_sq_sum = features.square().sum(dim=0)
        feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
        feature_sq_sum = (
            batch_sq_sum if feature_sq_sum is None else feature_sq_sum + batch_sq_sum
        )
        count += features.shape[0]

    mean = feature_sum / count
    variance = (feature_sq_sum / count - mean.square()).clamp_min(config.EPSILON)
    return {"mean": mean.float().cpu(), "variance": variance.float().cpu()}


def gaussian_kernel(sigma, device, dtype):
    radius = max(1, math.ceil(3 * sigma))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coords.square()) / (2 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)


def smooth_maps(maps):
    kernel = gaussian_kernel(config.GAUSSIAN_SIGMA, maps.device, maps.dtype)
    radius = kernel.shape[0] // 2
    return F.conv2d(
        maps[:, None],
        kernel[None, None],
        padding=radius,
    )[:, 0]


def aggregate_image_scores(maps):
    flat = maps.flatten(1)
    k = max(1, round(flat.shape[1] * config.IMAGE_TOP_FRACTION))
    return flat.topk(k, dim=1).values.mean(dim=1)


@torch.inference_mode()
def score_features(features, normal_model):
    mean = normal_model["mean"].to(features.device)
    variance = normal_model["variance"].to(features.device)
    # Diagonal Mahalanobis distance at each feature-map location.
    maps = ((features - mean).square() / variance).mean(dim=1).sqrt()
    maps = smooth_maps(maps)
    return maps, aggregate_image_scores(maps)


@torch.inference_mode()
def calibrate_model(model, loader, normal_model, device):
    all_maps = []
    all_scores = []
    for images, _ in loader:
        features = extract_features(model, images, device)
        maps, scores = score_features(features, normal_model)
        all_maps.append(maps.cpu())
        all_scores.append(scores.cpu())

    pixels = torch.cat(all_maps).flatten()
    image_scores = torch.cat(all_scores)
    normal_model["pixel_offset"] = torch.quantile(pixels, 0.50).item()
    normal_model["pixel_scale"] = max(
        torch.quantile(pixels, 0.995).item() - normal_model["pixel_offset"],
        config.EPSILON,
    )
    normal_model["image_offset"] = torch.quantile(image_scores, 0.50).item()
    normal_model["image_scale"] = max(
        torch.quantile(image_scores, 0.995).item() - normal_model["image_offset"],
        config.EPSILON,
    )
    return normal_model


def normalize_scores(values, offset, scale):
    # Monotonic robust scaling to [0, 1), preserving ranking for large outliers.
    excess = torch.clamp(values - offset, min=0)
    return excess / (excess + scale)


def save_pixel_map(anomaly_map, path, normal_model):
    normalized = normalize_scores(
        anomaly_map,
        normal_model["pixel_offset"],
        normal_model["pixel_scale"],
    )
    resized = F.interpolate(
        normalized[None, None],
        size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    array = (resized.clamp(0, 1) * 255).round().byte().cpu().numpy()
    Image.fromarray(array, mode="L").save(path)

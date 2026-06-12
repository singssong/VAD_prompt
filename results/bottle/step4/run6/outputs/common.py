from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18

import config


def list_images(directory: Path):
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and not p.name.startswith(".")
        and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms(
            crop_size=config.IMAGE_SIZE,
            resize_size=config.IMAGE_SIZE,
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB").resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            tensor = self.transform(image)
        return tensor, self.paths[index].name


class FeatureExtractor(nn.Module):
    """ImageNet ResNet-18 truncated after layer2."""

    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2

    def forward(self, images):
        stem = self.stem(images)
        level1 = self.layer1(stem)
        level2 = self.layer2(level1)
        level1 = F.adaptive_avg_pool2d(level1, level2.shape[-2:])
        features = torch.cat((level1, level2), dim=1)
        return F.normalize(features, p=2, dim=1)


@torch.inference_mode()
def extract_features(model, images):
    """Extract and concatenate normalized mid-level feature maps."""
    return model(images)


def build_normal_model(feature_batches, calibration_batches):
    """Store a sampled normal patch bank and normal-only score calibration."""
    patches = torch.cat(
        [batch.permute(0, 2, 3, 1).reshape(-1, batch.shape[1])
         for batch in feature_batches],
        dim=0,
    )
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    if len(patches) > config.MEMORY_BANK_SIZE:
        indices = torch.randperm(len(patches), generator=generator)[
            :config.MEMORY_BANK_SIZE
        ]
        patches = patches[indices]

    model_data = {"memory_bank": patches.contiguous().cpu()}
    calibration_maps = [
        nearest_neighbor_map(batch, model_data["memory_bank"])
        for batch in calibration_batches
    ]
    calibration_pixels = torch.cat([m.flatten() for m in calibration_maps])
    calibration_images = torch.cat(
        [aggregate_image_scores(m) for m in calibration_maps]
    )
    model_data["pixel_scale"] = robust_upper_scale(calibration_pixels)
    model_data["image_scale"] = robust_upper_scale(calibration_images)
    return model_data


def nearest_neighbor_map(features, memory_bank, query_chunk_size=1024):
    """Return the nearest normal-patch distance at every feature-map location."""
    device = features.device
    bank = memory_bank.to(device)
    batch, channels, height, width = features.shape
    queries = features.permute(0, 2, 3, 1).reshape(-1, channels)
    bank_sq = (bank * bank).sum(dim=1).unsqueeze(0)
    distances = []
    for chunk in queries.split(query_chunk_size):
        squared = (
            (chunk * chunk).sum(dim=1, keepdim=True)
            + bank_sq
            - 2.0 * chunk @ bank.T
        ).clamp_min_(0)
        distances.append(squared.min(dim=1).values.sqrt())
    return torch.cat(distances).reshape(batch, height, width).cpu()


def gaussian_smooth(maps, sigma):
    radius = max(1, int(round(3 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, dtype=maps.dtype)
    kernel = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    kernel2d = torch.outer(kernel, kernel)[None, None]
    padded = F.pad(
        maps[:, None], (radius, radius, radius, radius), mode="reflect"
    )
    return F.conv2d(padded, kernel2d).squeeze(1)


def aggregate_image_scores(maps):
    flat = maps.flatten(1)
    count = max(1, int(flat.shape[1] * (1.0 - config.IMAGE_SCORE_PERCENTILE)))
    return flat.topk(count, dim=1).values.mean(dim=1)


def robust_upper_scale(values):
    scale = torch.quantile(values.float(), 0.995).item()
    return max(scale, 1e-6)


def normalize_scores(values, scale):
    """Map nonnegative distances monotonically to the stable [0, 1) range."""
    values = torch.as_tensor(values, dtype=torch.float32).clamp_min(0)
    return values / (values + float(scale))


def score_features(features, model_data):
    """Compute smoothed pixel maps and normalized image anomaly scores."""
    maps = nearest_neighbor_map(features, model_data["memory_bank"])
    maps = gaussian_smooth(maps, config.GAUSSIAN_SIGMA)
    image_raw = aggregate_image_scores(maps)
    image_scores = normalize_scores(image_raw, model_data["image_scale"])
    pixel_scores = normalize_scores(maps, model_data["pixel_scale"])
    pixel_scores = F.interpolate(
        pixel_scores[:, None],
        size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    return pixel_scores.numpy(), image_scores.numpy()


def save_pixel_map(path, pixel_map):
    image = Image.fromarray(
        np.rint(np.clip(pixel_map, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    )
    image.save(path)

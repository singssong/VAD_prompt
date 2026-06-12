#!/usr/bin/env python3
"""Train a one-class, feature-based spatial nearest-neighbor model."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


# Configuration
ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "train_dir": ROOT / "data" / "train",
    "test_dir": ROOT / "data" / "test_images",
    "output_dir": ROOT / "outputs",
    "model_path": ROOT / "outputs" / "model.pt",
    "image_size": 256,
    "layers": ("layer2", "layer3"),
    "descriptor_size": 32,
    "projection_dim": 128,
    "batch_size": 8,
    "num_workers": 2,
    "gaussian_sigma": 1.25,
    "top_fraction": 0.01,
    "seed": 17,
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(directory: Path) -> list[Path]:
    """Return all supported images without inspecting any other directory."""
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, files: list[Path], image_size: int) -> None:
        self.files = files
        self.image_size = image_size
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        self.normalize = weights.transforms(
            crop_size=image_size,
            resize_size=image_size,
            antialias=True,
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.files[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            tensor = self.normalize(image)
        return tensor, path.name


class FeatureExtractor(nn.Module):
    """ImageNet backbone exposing two mid-level feature maps."""

    def __init__(self) -> None:
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stem(images)
        features = self.layer1(features)
        level2 = self.layer2(features)
        level3 = self.layer3(level2)
        return level2, level3


def make_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    """Create a deterministic orthonormal random projection."""
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator)
    projection, _ = torch.linalg.qr(matrix, mode="reduced")
    return projection


def extract_features(
    extractor: FeatureExtractor,
    images: torch.Tensor,
    projection: torch.Tensor,
    descriptor_size: int,
) -> torch.Tensor:
    """Extract, align, concatenate, and project two feature depths."""
    level2, level3 = extractor(images)
    level2 = F.interpolate(
        level2, size=(descriptor_size, descriptor_size), mode="bilinear", align_corners=False
    )
    level3 = F.interpolate(
        level3, size=(descriptor_size, descriptor_size), mode="bilinear", align_corners=False
    )
    level2 = F.normalize(level2, dim=1)
    level3 = F.normalize(level3, dim=1)
    features = torch.cat((level2, level3), dim=1).permute(0, 2, 3, 1)
    features = torch.matmul(features, projection)
    return F.normalize(features, dim=-1)


def build_normal_model(
    extractor: FeatureExtractor,
    loader: DataLoader,
    projection: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Store projected normal descriptors as [positions, images, channels]."""
    batches = []
    with torch.inference_mode():
        for images, _ in loader:
            descriptors = extract_features(
                extractor,
                images.to(device),
                projection,
                CONFIG["descriptor_size"],
            )
            batches.append(descriptors.cpu())
    bank = torch.cat(batches).flatten(1, 2).transpose(0, 1).contiguous()
    return bank.to(torch.float16)


def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, math.ceil(3 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return torch.outer(kernel, kernel)[None, None]


def smooth_maps(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel(sigma, maps.device)
    padding = kernel.shape[-1] // 2
    return F.conv2d(maps[:, None], kernel, padding=padding)[:, 0]


def aggregate_maps(maps: torch.Tensor, top_fraction: float) -> torch.Tensor:
    count = max(1, round(maps.shape[-1] * maps.shape[-2] * top_fraction))
    return maps.flatten(1).topk(count, dim=1).values.mean(dim=1)


def score_descriptors(
    descriptors: torch.Tensor,
    normal_bank: torch.Tensor,
    position_chunk: int = 64,
) -> torch.Tensor:
    """Return nearest-normal distance at every descriptor-grid position."""
    batch_size, height, width, channels = descriptors.shape
    queries = descriptors.flatten(1, 2).transpose(0, 1)
    scores = []
    for start in range(0, queries.shape[0], position_chunk):
        query = queries[start : start + position_chunk].float()
        reference = normal_bank[start : start + position_chunk].to(
            device=query.device, dtype=torch.float32
        )
        distances = torch.cdist(query, reference)
        scores.append(distances.min(dim=-1).values)
    return torch.cat(scores).transpose(0, 1).reshape(batch_size, height, width)


def leave_one_out_scores(
    normal_bank: torch.Tensor, device: torch.device, position_chunk: int = 16
) -> torch.Tensor:
    """Score each normal image against all other normal images."""
    position_count, image_count, _ = normal_bank.shape
    result = torch.empty(image_count, position_count)
    diagonal = torch.arange(image_count, device=device)
    for start in range(0, position_count, position_chunk):
        reference = normal_bank[start : start + position_chunk].to(
            device=device, dtype=torch.float32
        )
        distances = torch.cdist(reference, reference)
        distances[:, diagonal, diagonal] = float("inf")
        result[:, start : start + position_chunk] = distances.min(dim=-1).values.T.cpu()
    side = round(math.sqrt(position_count))
    return result.reshape(image_count, side, side)


def fit_calibration(raw_maps: torch.Tensor) -> dict[str, float]:
    """Fit robust score scaling from normal-only leave-one-out scores."""
    smoothed = smooth_maps(raw_maps, CONFIG["gaussian_sigma"])
    image_scores = aggregate_maps(smoothed, CONFIG["top_fraction"])
    pixel_values = smoothed.flatten()
    image_low = torch.quantile(image_scores, 0.05).item()
    image_high = torch.quantile(image_scores, 0.99).item()
    pixel_low = torch.quantile(pixel_values, 0.50).item()
    pixel_high = torch.quantile(pixel_values, 0.999).item()
    return {
        "image_low": image_low,
        "image_scale": max(image_high - image_low, 1e-6),
        "pixel_low": pixel_low,
        "pixel_scale": max(pixel_high - pixel_low, 1e-6),
    }


def main() -> None:
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.backends.cudnn.benchmark = True
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    files = image_files(CONFIG["train_dir"])
    if not files:
        raise RuntimeError(f"No training images found in {CONFIG['train_dir']}")
    loader = DataLoader(
        ImageDataset(files, CONFIG["image_size"]),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor().eval().to(device)
    projection = make_projection(1536, CONFIG["projection_dim"], CONFIG["seed"]).to(device)

    print(f"Extracting normal features from {len(files)} images on {device}...")
    normal_bank = build_normal_model(extractor, loader, projection, device)
    print("Computing leave-one-out normal calibration...")
    normal_maps = leave_one_out_scores(normal_bank, device)
    calibration = fit_calibration(normal_maps)

    torch.save(
        {
            "normal_bank": normal_bank,
            "projection": projection.cpu(),
            "calibration": calibration,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in CONFIG.items()
            },
            "backbone": "wide_resnet50_2",
            "weights": "IMAGENET1K_V2",
        },
        CONFIG["model_path"],
    )
    print(f"Saved model to {CONFIG['model_path']}")


if __name__ == "__main__":
    main()

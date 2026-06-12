from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


# Central configuration for both training and inference.
ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "normal_feature_model.pt"

IMAGE_SIZE = 256
BACKBONE = "wide_resnet50_2"
FEATURE_LAYERS = ("layer2", "layer3")
EMBEDDING_DIM = 128
MEMORY_SIZE = 40000
CALIBRATION_FRACTION = 0.10
GAUSSIAN_SIGMA = 2.0
TOP_FRACTION = 0.01
SEED = 17
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((tensor - mean) / std).to(device)


class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(images)
        x = self.layer1(x)
        middle = self.layer2(x)
        deep = self.layer3(middle)
        return middle, deep


@torch.inference_mode()
def extract_features(
    extractor: FeatureExtractor,
    images: torch.Tensor,
    projection: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int]]:
    middle, deep = extractor(images)
    deep = F.interpolate(deep, size=middle.shape[-2:], mode="bilinear", align_corners=False)
    middle = F.normalize(middle, dim=1)
    deep = F.normalize(deep, dim=1)
    combined = torch.cat((middle, deep), dim=1)
    height, width = combined.shape[-2:]
    patches = combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1])
    patches = patches @ projection
    return F.normalize(patches, dim=1), (height, width)


def make_projection(input_dim: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(SEED)
    projection = torch.randn(
        input_dim, EMBEDDING_DIM, generator=generator, device=device
    )
    return projection / math.sqrt(EMBEDDING_DIM)


def build_normal_model(
    extractor: FeatureExtractor,
    train_paths: list[Path],
    device: torch.device,
) -> dict:
    projection = make_projection(1536, device)
    shuffled = train_paths.copy()
    random.Random(SEED).shuffle(shuffled)
    calibration_count = max(1, round(len(shuffled) * CALIBRATION_FRACTION))
    calibration_paths = shuffled[:calibration_count]
    bank_paths = shuffled[calibration_count:]

    bank_parts = []
    for path in bank_paths:
        features, _ = extract_features(extractor, load_image(path, device), projection)
        bank_parts.append(features.cpu())
    all_bank = torch.cat(bank_parts)
    generator = torch.Generator().manual_seed(SEED)
    selected = torch.randperm(len(all_bank), generator=generator)[:MEMORY_SIZE]
    memory_bank = all_bank[selected].contiguous().to(device)

    calibration_maps = []
    calibration_image_scores = []
    for path in calibration_paths:
        features, grid_size = extract_features(
            extractor, load_image(path, device), projection
        )
        anomaly_map = score_feature_grid(features, grid_size, memory_bank)
        calibration_maps.append(anomaly_map.flatten().cpu())
        calibration_image_scores.append(aggregate_image_score(anomaly_map))

    pixel_values = torch.cat(calibration_maps).numpy()
    image_values = np.asarray(calibration_image_scores, dtype=np.float32)
    image_center = float(np.median(image_values))
    image_mad = float(np.median(np.abs(image_values - image_center)))
    image_scale = max(1.4826 * image_mad, float(np.std(image_values)) * 0.25, 1e-6)

    return {
        "backbone": BACKBONE,
        "feature_layers": FEATURE_LAYERS,
        "image_size": IMAGE_SIZE,
        "projection": projection.cpu(),
        "memory_bank": memory_bank.cpu(),
        "pixel_low": float(np.quantile(pixel_values, 0.50)),
        "pixel_high": float(np.quantile(pixel_values, 0.999)),
        "image_center": image_center,
        "image_scale": image_scale,
        "calibration_files": [path.name for path in calibration_paths],
    }


@torch.inference_mode()
def nearest_neighbor_distances(
    features: torch.Tensor,
    memory_bank: torch.Tensor,
    chunk_size: int = 8192,
) -> torch.Tensor:
    best_similarities = []
    for start in range(0, len(features), chunk_size):
        similarities = features[start:start + chunk_size] @ memory_bank.T
        best_similarities.append(similarities.max(dim=1).values)
    return (1.0 - torch.cat(best_similarities)).clamp_min(0.0)


def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def gaussian_smooth(anomaly_map: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel(sigma, anomaly_map.device)
    radius = kernel.numel() // 2
    data = anomaly_map[None, None]
    horizontal = kernel.view(1, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1)
    data = F.pad(data, (radius, radius, 0, 0), mode="reflect")
    data = F.conv2d(data, horizontal)
    data = F.pad(data, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(data, vertical)[0, 0]


def score_feature_grid(
    features: torch.Tensor,
    grid_size: tuple[int, int],
    memory_bank: torch.Tensor,
) -> torch.Tensor:
    distances = nearest_neighbor_distances(features, memory_bank)
    anomaly_map = distances.reshape(grid_size)
    anomaly_map = gaussian_smooth(anomaly_map, GAUSSIAN_SIGMA)
    return F.interpolate(
        anomaly_map[None, None],
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def aggregate_image_score(anomaly_map: torch.Tensor) -> float:
    count = max(1, round(anomaly_map.numel() * TOP_FRACTION))
    return float(torch.topk(anomaly_map.flatten(), count).values.mean().item())


def normalize_image_score(raw_score: float, center: float, scale: float) -> float:
    # Shifted robust sigmoid maps normal calibration scores near zero while
    # retaining ordering and avoiding hard saturation for strong anomalies.
    value = (raw_score - center) / scale - 3.0
    value = max(-60.0, min(60.0, value))
    return float(1.0 / (1.0 + math.exp(-value)))


def save_pixel_map(
    anomaly_map: torch.Tensor,
    destination: Path,
    pixel_low: float,
    pixel_high: float,
) -> None:
    denominator = max(pixel_high - pixel_low, 1e-8)
    normalized = ((anomaly_map - pixel_low) / denominator).clamp(0.0, 1.0)
    array = (normalized.cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="L").save(destination)


@torch.inference_mode()
def score_images(
    extractor: FeatureExtractor,
    model: dict,
    paths: Iterable[Path],
    device: torch.device,
) -> dict[str, float]:
    projection = model["projection"].to(device)
    memory_bank = model["memory_bank"].to(device)
    scores = {}
    PIXEL_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        features, grid_size = extract_features(
            extractor, load_image(path, device), projection
        )
        anomaly_map = score_feature_grid(features, grid_size, memory_bank)
        raw_score = aggregate_image_score(anomaly_map)
        scores[path.name] = normalize_image_score(
            raw_score, model["image_center"], model["image_scale"]
        )
        save_pixel_map(
            anomaly_map,
            PIXEL_DIR / path.name,
            model["pixel_low"],
            model["pixel_high"],
        )
    return scores

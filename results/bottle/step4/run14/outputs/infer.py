"""Score test images using the trained feature-statistics model."""

from pathlib import Path
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.feature_extraction import create_feature_extractor


# Configuration
ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "model.pt"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"
IMAGE_SIZE = 256
LAYERS = {"layer2": "mid", "layer3": "deep"}
BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = ResNet50_Weights.IMAGENET1K_V2.transforms(
            crop_size=IMAGE_SIZE, resize_size=IMAGE_SIZE
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            return self.transform(image), path.name


def build_feature_extractor(device):
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    extractor = create_feature_extractor(backbone, return_nodes=LAYERS)
    return extractor.eval().to(device)


def extract_features(extractor, images):
    outputs = extractor(images)
    mid = outputs["mid"]
    deep = F.interpolate(
        outputs["deep"], size=mid.shape[-2:], mode="bilinear", align_corners=False
    )
    return torch.cat((mid, deep), dim=1)


def gaussian_kernel(sigma, device):
    radius = max(1, math.ceil(3 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)[None, None]


def score_feature_maps(features, mean, inv_std, sigma):
    standardized = (features - mean) * inv_std
    maps = standardized.square().mean(dim=1, keepdim=True)
    kernel = gaussian_kernel(sigma, maps.device)
    return F.conv2d(maps, kernel, padding=kernel.shape[-1] // 2)


def aggregate_image_scores(pixel_maps, top_fraction):
    flat = pixel_maps.flatten(1)
    top_count = max(1, math.ceil(flat.shape[1] * top_fraction))
    return flat.topk(top_count, dim=1).values.mean(dim=1)


def normalize_scores(values, center, scale):
    """Map calibrated scores monotonically into a stable [0, 1] range."""
    log_center = math.log1p(center)
    log_scale = max(math.log1p(center + scale) - log_center, 1e-6)
    z = (torch.log1p(values) - log_center) / log_scale
    return (0.5 + torch.atan(z) / math.pi).clamp(0, 1)


def save_pixel_map(pixel_map, path):
    array = (pixel_map.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def score_images(extractor, loader, model, device):
    config = model["config"]
    calibration = model["calibration"]
    mean = model["mean"].to(device)
    inv_std = model["inv_std"].to(device)
    results = {}
    PIXEL_DIR.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for images, names in loader:
            features = extract_features(extractor, images.to(device))
            maps = score_feature_maps(
                features, mean, inv_std, config["gaussian_sigma"]
            )
            raw_image_scores = aggregate_image_scores(
                maps, config["top_fraction"]
            )
            image_scores = normalize_scores(
                raw_image_scores,
                calibration["image_center"],
                calibration["image_scale"],
            )
            normalized_maps = normalize_scores(
                maps,
                calibration["pixel_center"],
                calibration["pixel_scale"],
            )
            normalized_maps = F.interpolate(
                normalized_maps, size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear", align_corners=False
            )

            for index, name in enumerate(names):
                results[name] = float(image_scores[index].cpu())
                save_pixel_map(normalized_maps[index, 0], PIXEL_DIR / name)
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if model["config"]["image_size"] != IMAGE_SIZE:
        raise RuntimeError("Inference configuration does not match trained model")
    dataset = ImageDataset(TEST_DIR)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda"
    )
    extractor = build_feature_extractor(device)
    scores = score_images(extractor, loader, model, device)
    with SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images; wrote {SCORES_PATH}")


if __name__ == "__main__":
    main()

import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

import config
from train import (
    ImageDataset,
    build_feature_extractor,
    extract_features,
    list_images,
    nearest_neighbor_distances,
)


def gaussian_kernel(sigma, device):
    radius = max(1, math.ceil(3 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)[None, None]


def smooth_and_resize(anomaly_map):
    kernel = gaussian_kernel(config.GAUSSIAN_SIGMA, anomaly_map.device)
    padding = kernel.shape[-1] // 2
    anomaly_map = F.pad(anomaly_map, (padding,) * 4, mode="reflect")
    anomaly_map = F.conv2d(anomaly_map, kernel)
    return F.interpolate(
        anomaly_map,
        size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )


def normalize_image_score(raw_score, calibration):
    z = (raw_score - calibration["image_center"]) / calibration["image_scale"]
    # A smooth bounded score is stable across runs and does not depend on test-set extrema.
    return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))


def normalize_pixel_map(anomaly_map, calibration):
    low = calibration["patch_low"]
    high = calibration["patch_high"]
    return ((anomaly_map - low) / (high - low)).clamp(0.0, 1.0)


@torch.inference_mode()
def score_images(extractor, paths, artifact, device):
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    projection = artifact["projection"].float().to(device)
    memory_bank = artifact["memory_bank"].float().to(device)
    calibration = artifact["calibration"]
    image_scores = {}

    config.PIXEL_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    for images, names in loader:
        features = extract_features(
            extractor, images.to(device, non_blocking=True), projection
        )
        height, width = features.shape[1:3]
        distances = nearest_neighbor_distances(
            features.reshape(-1, config.EMBEDDING_DIM), memory_bank
        )
        top_count = max(1, round(distances.numel() * config.TOP_FRACTION))
        raw_image_score = distances.topk(top_count).values.mean().item()
        image_scores[names[0]] = normalize_image_score(raw_image_score, calibration)

        anomaly_map = distances.reshape(1, 1, height, width)
        anomaly_map = smooth_and_resize(anomaly_map)
        anomaly_map = normalize_pixel_map(anomaly_map, calibration)
        output = (anomaly_map[0, 0].cpu().numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(output, mode="L").save(config.PIXEL_SCORES_DIR / names[0])

    return image_scores


def main():
    paths = list_images(config.TEST_DIR)
    if not paths:
        raise RuntimeError(f"No test images found in {config.TEST_DIR}")
    if not config.MODEL_PATH.exists():
        raise RuntimeError(f"Missing model artifact. Run train.py first: {config.MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    extractor = build_feature_extractor(device)
    scores = score_images(extractor, paths, artifact, device)
    with config.IMAGE_SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images")


if __name__ == "__main__":
    main()

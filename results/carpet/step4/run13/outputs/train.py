from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from anomaly import (
    FeatureExtractor,
    ImageDataset,
    build_normal_model,
    create_projection,
    extract_features,
    list_images,
    robust_range,
    score_features,
)
from config import CONFIG


def train() -> None:
    config = CONFIG
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    paths = list_images(config.train_dir)
    if not paths:
        raise RuntimeError(f"No training images found in {config.train_dir}")

    indices = list(range(len(paths)))
    random.Random(config.seed).shuffle(indices)
    calibration_count = max(1, round(len(paths) * config.calibration_fraction))
    calibration_paths = [paths[index] for index in indices[:calibration_count]]
    bank_paths = [paths[index] for index in indices[calibration_count:]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor().to(device)
    projection = create_projection(1536, config.embedding_dim, config.seed, device)

    bank_loader = DataLoader(
        ImageDataset(bank_paths, config.image_size),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    patch_batches = []
    for images, _ in bank_loader:
        patches, _ = extract_features(
            extractor, images.to(device, non_blocking=True), projection
        )
        patch_batches.append(patches.reshape(-1, config.embedding_dim).cpu())
    memory_bank_cpu = build_normal_model(
        patch_batches, config.memory_bank_size, config.seed
    )
    memory_bank = memory_bank_cpu.to(device)

    calibration_loader = DataLoader(
        ImageDataset(calibration_paths, config.image_size),
        batch_size=config.inference_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    calibration_maps = []
    calibration_scores = []
    for images, _ in calibration_loader:
        patches, grid_size = extract_features(
            extractor, images.to(device, non_blocking=True), projection
        )
        maps, scores = score_features(patches, grid_size, memory_bank, config)
        calibration_maps.append(maps.cpu())
        calibration_scores.append(scores.cpu())

    pixel_low, pixel_high = robust_range(
        torch.cat(calibration_maps), 0.50, 0.995
    )
    image_low, image_high = robust_range(
        torch.cat(calibration_scores), 0.10, 0.995
    )
    checkpoint = {
        "method": "PatchCore-style projected patch nearest neighbors",
        "backbone": "wide_resnet50_2 (ImageNet1K V2)",
        "feature_layers": config.feature_layers,
        "projection": projection.cpu(),
        "memory_bank": memory_bank_cpu,
        "pixel_range": (pixel_low, pixel_high),
        "image_range": (image_low, image_high),
        "grid_size": grid_size,
        "image_size": config.image_size,
        "seed": config.seed,
    }
    torch.save(checkpoint, config.model_path)
    print(
        f"Saved {len(memory_bank_cpu)} normal patch embeddings to "
        f"{config.model_path}"
    )


if __name__ == "__main__":
    train()

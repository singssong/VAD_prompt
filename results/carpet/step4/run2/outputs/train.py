#!/usr/bin/env python3
import torch

import config
from anomaly_pipeline import (
    FeatureExtractor,
    build_normal_feature_model,
    estimate_calibration,
    list_images,
    set_deterministic,
)


def main():
    set_deterministic(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_paths = list_images(config.TRAIN_DIR)
    print(f"Training on {len(train_paths)} normal images with {device}.")

    extractor = FeatureExtractor().to(device)
    memory_bank, projection, calibration_paths = build_normal_feature_model(
        extractor, train_paths, device
    )
    calibration = estimate_calibration(
        extractor, calibration_paths, memory_bank, projection, device
    )
    artifact = {
        "method": "PatchCore-style projected nearest-neighbor patch features",
        "backbone": config.BACKBONE,
        "feature_layers": config.FEATURE_LAYERS,
        "image_size": config.IMAGE_SIZE,
        "memory_bank": memory_bank.half(),
        "projection": projection,
        "calibration": calibration,
    }
    torch.save(artifact, config.MODEL_PATH)
    print(
        f"Saved {len(memory_bank)} normal patches and calibration to "
        f"{config.MODEL_PATH}."
    )


if __name__ == "__main__":
    main()

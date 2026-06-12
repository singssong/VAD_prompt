#!/usr/bin/env python3
import torch

import config
from anomaly import (
    FeatureExtractor,
    build_normal_model,
    image_files,
    make_loader,
    robust_calibration,
    score_loader,
    seed_everything,
)


def main() -> None:
    seed_everything(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_paths = image_files(config.TRAIN_DIR)
    if not train_paths:
        raise RuntimeError(f"No training images found in {config.TRAIN_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor().to(device)
    loader = make_loader(train_paths)

    print(f"Extracting normal features from {len(train_paths)} images on {device}...")
    memory_bank = build_normal_model(extractor, loader, device)
    print(f"Built memory bank with {len(memory_bank)} patch features.")

    # Score normal training images to derive dataset-specific robust normalization.
    _, image_scores, pixel_maps = score_loader(
        extractor, memory_bank, loader, device
    )
    checkpoint = {
        "method": "PatchCore-style nearest-neighbor feature memory",
        "backbone": config.BACKBONE,
        "feature_layers": config.FEATURE_LAYERS,
        "image_size": config.IMAGE_SIZE,
        "memory_bank": memory_bank,
        "image_calibration": robust_calibration(image_scores),
        "pixel_calibration": robust_calibration(pixel_maps),
    }
    torch.save(checkpoint, config.MODEL_PATH)
    print(f"Saved model to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()

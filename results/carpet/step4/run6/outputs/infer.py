#!/usr/bin/env python3
import json

import numpy as np
import torch
from PIL import Image

import config
from anomaly import (
    FeatureExtractor,
    image_files,
    make_loader,
    normalize_scores,
    score_loader,
    seed_everything,
)


def main() -> None:
    seed_everything(config.SEED)
    test_paths = image_files(config.TEST_DIR)
    if not test_paths:
        raise RuntimeError(f"No test images found in {config.TEST_DIR}")
    if not config.MODEL_PATH.exists():
        raise RuntimeError("Model missing. Run outputs/train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    extractor = FeatureExtractor().to(device)
    names, raw_image_scores, raw_pixel_maps = score_loader(
        extractor,
        checkpoint["memory_bank"],
        make_loader(test_paths),
        device,
    )

    image_scores = normalize_scores(
        raw_image_scores, checkpoint["image_calibration"]
    )
    pixel_maps = normalize_scores(
        raw_pixel_maps, checkpoint["pixel_calibration"]
    )

    config.PIXEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, pixel_map in zip(names, pixel_maps):
        pixels = (
            pixel_map.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
        )
        Image.fromarray(np.asarray(pixels), mode="L").save(
            config.PIXEL_OUTPUT_DIR / name
        )

    result = {
        name: float(score)
        for name, score in zip(names, image_scores.tolist())
    }
    with config.SCORES_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    print(f"Scored {len(result)} images and wrote results to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

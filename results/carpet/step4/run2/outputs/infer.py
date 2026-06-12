#!/usr/bin/env python3
import json

import numpy as np
import torch
from PIL import Image

import config
from anomaly_pipeline import (
    FeatureExtractor,
    list_images,
    normalize_image_score,
    normalize_score,
    score_images,
    set_deterministic,
)


def main():
    set_deterministic(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    test_paths = list_images(config.TEST_DIR)
    config.PIXEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor().to(device)
    memory_bank = artifact["memory_bank"].float()
    projection = artifact["projection"].float()
    calibration = artifact["calibration"]
    scores = {}

    for names, pixel_maps, image_scores in score_images(
        extractor, test_paths, memory_bank, projection, device
    ):
        normalized_maps = normalize_score(
            pixel_maps, calibration["pixel_low"], calibration["pixel_high"]
        )
        for name, pixel_map, image_score in zip(
            names, normalized_maps, image_scores.tolist()
        ):
            output = np.rint(pixel_map.numpy() * 255.0).astype(np.uint8)
            Image.fromarray(output, mode="L").save(config.PIXEL_OUTPUT_DIR / name)
            scores[name] = normalize_image_score(
                image_score,
                calibration["image_low"],
                calibration["image_high"],
            )

    with config.IMAGE_SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images with {device}.")


if __name__ == "__main__":
    main()

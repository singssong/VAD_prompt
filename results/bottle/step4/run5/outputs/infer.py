#!/usr/bin/env python3
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import config
from pipeline import (
    ImageDataset,
    MidLevelResNet18,
    aggregate_image_scores,
    extract_features,
    normalize_scores,
    resize_maps,
    score_feature_map,
    smooth_maps,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset(config.TEST_DIR)
    if not dataset.paths:
        raise RuntimeError(f"No test images found in {config.TEST_DIR}")

    state = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    mean = state["mean"].to(device)
    variance = state["variance"].to(device)
    calibration = state["calibration"]
    model = MidLevelResNet18().eval().to(device)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    config.PIXEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_scores = {}
    with torch.inference_mode():
        for images, filenames in loader:
            features = extract_features(model, images.to(device))
            maps = smooth_maps(score_feature_map(features, mean, variance))
            maps = resize_maps(maps)
            raw_image_scores = aggregate_image_scores(maps)
            normalized_images = normalize_scores(
                raw_image_scores,
                calibration["image_low"],
                calibration["image_high"],
            )
            normalized_maps = normalize_scores(
                maps,
                calibration["pixel_low"],
                calibration["pixel_high"],
            )

            for filename, score, anomaly_map in zip(
                filenames, normalized_images, normalized_maps
            ):
                image_scores[filename] = float(score.cpu())
                pixels = (
                    anomaly_map.mul(255).round().byte().cpu().numpy().astype(np.uint8)
                )
                Image.fromarray(pixels, mode="L").save(
                    config.PIXEL_OUTPUT_DIR / filename
                )

    with config.SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(image_scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(image_scores)} images; results saved in {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()


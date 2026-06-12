"""Score all test images and write image-level and pixel-level outputs."""

import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import config
from train import (
    FeatureExtractor,
    ImageDataset,
    aggregate_image_scores,
    extract_features,
    score_features,
)


def normalize(values, low, high):
    return ((values - low) / max(high - low, 1e-8)).clamp(0.0, 1.0)


def score_images(extractor, loader, model, device):
    """Scoring stage: feature maps to normalized maps and image scores."""
    mean = model["mean"].to(device)
    variance = model["variance"].to(device)
    calibration = model["calibration"]
    results = {}
    config.PIXEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for images, names in loader:
        features = extract_features(extractor, images.to(device))
        raw_maps = score_features(features, mean, variance)
        raw_image_scores = aggregate_image_scores(raw_maps)
        normalized_maps = normalize(
            raw_maps, calibration["pixel_low"], calibration["pixel_high"]
        )
        normalized_scores = normalize(
            raw_image_scores, calibration["image_low"], calibration["image_high"]
        )

        for name, anomaly_map, score in zip(names, normalized_maps, normalized_scores):
            pixels = (anomaly_map.cpu().numpy() * 255.0).round().astype(np.uint8)
            Image.fromarray(pixels, mode="L").save(config.PIXEL_OUTPUT_DIR / name)
            results[name] = float(score.item())
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    dataset = ImageDataset(config.TEST_DIR)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().to(device)
    scores = score_images(extractor, loader, model, device)
    with config.SCORES_PATH.open("w", encoding="utf-8") as output:
        json.dump(scores, output, indent=2, sort_keys=True)
        output.write("\n")
    print(f"Scored {len(scores)} images and wrote {config.SCORES_PATH}")


if __name__ == "__main__":
    main()


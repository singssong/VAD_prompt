import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from model import ResNetFeatureExtractor, extract_features, list_images
from train import (
    aggregate_image_scores,
    make_loader,
    nearest_neighbor_distances,
    postprocess_maps,
)


def normalize_pixels(values, low, high):
    return ((values - low) / (high - low)).clamp(0.0, 1.0)


def normalize_image_scores(values, low, high):
    shifted = (values - low).clamp_min(0.0)
    scale = max(high - low, 1e-6)
    return shifted / (shifted + scale)


@torch.inference_mode()
def score_images(extractor, loader, projection, memory_bank, calibration, device):
    """Score test images and return normalized image and pixel anomaly scores."""
    results = {}
    pixel_maps = {}
    for images, names in loader:
        features = extract_features(
            extractor, images.to(device, non_blocking=True), projection
        )
        distances = nearest_neighbor_distances(features, memory_bank)
        maps = postprocess_maps(distances, images.shape[0])
        raw_image_scores = aggregate_image_scores(maps)
        image_scores = normalize_image_scores(
            raw_image_scores,
            calibration["image_low"],
            calibration["image_high"],
        )
        maps = normalize_pixels(
            maps, calibration["pixel_low"], calibration["pixel_high"]
        )
        maps = F.interpolate(
            maps,
            size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        for name, score, anomaly_map in zip(names, image_scores, maps):
            results[name] = float(score.cpu())
            pixel_maps[name] = anomaly_map[0].cpu().numpy()
    return results, pixel_maps


def save_outputs(image_scores, pixel_maps):
    config.PIXEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with config.SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(image_scores, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for name, anomaly_map in pixel_maps.items():
        pixels = np.rint(anomaly_map * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(pixels, mode="L").save(config.PIXEL_OUTPUT_DIR / name)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    paths = list_images(config.TEST_DIR)
    if not paths:
        raise RuntimeError(f"No test images found in {config.TEST_DIR}")

    extractor = ResNetFeatureExtractor().to(device)
    projection = artifact["projection"].to(device)
    memory_bank = artifact["memory_bank"].to(device)
    image_scores, pixel_maps = score_images(
        extractor,
        make_loader(paths),
        projection,
        memory_bank,
        artifact["calibration"],
        device,
    )
    save_outputs(image_scores, pixel_maps)
    print(f"Scored {len(image_scores)} images")
    print(f"Image scores: {config.SCORES_PATH}")
    print(f"Pixel maps: {config.PIXEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

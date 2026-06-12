import json

import torch
from PIL import Image

import config
from common import (
    FeatureExtractor,
    extract_feature_batches,
    list_images,
    robust_scale,
    score_feature_maps,
    seed_everything,
)


def save_pixel_map(path, anomaly_map):
    array = (anomaly_map * 255.0).round().to(torch.uint8).numpy()
    Image.fromarray(array, mode="L").save(path)


def score_test_images():
    seed_everything(config.SEED)
    test_paths = list_images(config.TEST_DIR)
    if not test_paths:
        raise RuntimeError(f"No test images found in {config.TEST_DIR}")

    artifact = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    memory_bank = artifact["memory_bank"].to(torch.float32)
    calibration = artifact["calibration"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor().to(device).eval()
    config.PIXEL_SCORE_DIR.mkdir(parents=True, exist_ok=True)
    scores = {}

    print(f"Scoring {len(test_paths)} images on {device}...")
    for features, names in extract_feature_batches(
        extractor, test_paths, device, config.INFER_BATCH_SIZE
    ):
        pixel_maps, image_scores = score_feature_maps(features, memory_bank, device)
        normalized_pixels = robust_scale(
            pixel_maps, calibration["pixel_low"], calibration["pixel_high"]
        )
        normalized_images = robust_scale(
            image_scores, calibration["image_low"], calibration["image_high"]
        )
        for name, pixel_map, image_score in zip(
            names, normalized_pixels, normalized_images
        ):
            save_pixel_map(config.PIXEL_SCORE_DIR / name, pixel_map)
            scores[name] = float(image_score)

    with config.IMAGE_SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote scores to {config.IMAGE_SCORES_PATH}")


if __name__ == "__main__":
    score_test_images()

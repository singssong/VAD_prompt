from __future__ import annotations

import json

import torch

import config
from pipeline import (
    FeatureExtractor,
    aggregate_image_scores,
    extract_features,
    list_images,
    make_loader,
    nearest_neighbor_maps,
    normalize_scores,
    postprocess_pixel_maps,
    save_grayscale_png,
)


def main() -> None:
    test_paths = list_images(config.TEST_DIR)
    if not test_paths:
        raise RuntimeError(f"No test images found in {config.TEST_DIR}")
    if not config.MODEL_PATH.exists():
        raise RuntimeError("Model not found. Run train.py first.")

    config.PIXEL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    extractor = FeatureExtractor().to(device)
    projection = model["projection"].to(device)
    memory_bank = model["memory_bank"].to(device)

    image_scores: dict[str, float] = {}
    loader = make_loader(test_paths, device)
    with torch.inference_mode():
        for images, filenames in loader:
            features = extract_features(extractor, images.to(device), projection)
            raw_maps = nearest_neighbor_maps(features, memory_bank)
            raw_scores = aggregate_image_scores(raw_maps)
            scores = normalize_scores(
                raw_scores,
                model["image_score_low"],
                model["image_score_high"],
            )
            pixel_maps = postprocess_pixel_maps(
                raw_maps,
                model["pixel_score_low"],
                model["pixel_score_high"],
            )
            for filename, score, pixel_map in zip(filenames, scores, pixel_maps):
                image_scores[filename] = float(score.cpu().item())
                save_grayscale_png(pixel_map, config.PIXEL_DIR / filename)

    with config.SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(image_scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(image_scores)} images and wrote {config.SCORES_PATH}")


if __name__ == "__main__":
    main()

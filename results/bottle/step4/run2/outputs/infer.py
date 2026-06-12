from __future__ import annotations

import json

import torch

import config
from pipeline import (
    FeatureExtractor,
    image_files,
    make_loader,
    normalize,
    normalize_image_scores,
    save_pixel_map,
    score_batch,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = image_files(config.TEST_DIR)
    if not paths:
        raise RuntimeError(f"No test images in {config.TEST_DIR}")

    model = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    calibration = model["calibration"]
    extractor = FeatureExtractor().to(device)
    config.PIXEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, float] = {}
    with torch.inference_mode():
        for images, names in make_loader(paths):
            maps, raw_scores = score_batch(
                extractor, images.to(device, non_blocking=True), model, device
            )
            scores = normalize_image_scores(
                raw_scores,
                calibration["image_low"],
                calibration["image_high"],
            )
            normalized_maps = normalize(
                maps,
                calibration["pixel_low"],
                calibration["pixel_high"],
            )
            for name, score, pixel_map in zip(names, scores, normalized_maps):
                results[name] = float(score)
                save_pixel_map(config.PIXEL_OUTPUT_DIR / name, pixel_map)

    with config.SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(results.items())), handle, indent=2)
        handle.write("\n")
    print(f"Scored {len(results)} images and wrote {config.SCORES_PATH}")


if __name__ == "__main__":
    main()

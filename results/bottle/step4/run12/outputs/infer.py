from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

import config
from pipeline import (
    FeatureExtractor,
    list_images,
    make_loader,
    minmax_normalize,
    score_images,
)


def main() -> None:
    test_paths = list_images(config.TEST_DIR)
    if not test_paths:
        raise RuntimeError(f"No test images found in {config.TEST_DIR}")
    if not config.MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: run train.py first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_model = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    memory_bank = saved_model["memory_bank"].to(device)
    extractor = FeatureExtractor().to(device)

    names: list[str] = []
    raw_maps: list[torch.Tensor] = []
    raw_scores: list[torch.Tensor] = []
    for images, batch_names in make_loader(test_paths):
        maps, scores = score_images(extractor, images, memory_bank, device)
        names.extend(batch_names)
        raw_maps.append(maps.cpu())
        raw_scores.append(scores.cpu())

    maps = torch.cat(raw_maps)
    scores = minmax_normalize(torch.cat(raw_scores))

    # A global robust scale preserves relative anomaly intensity across images.
    map_low = torch.quantile(maps, 0.01)
    map_high = torch.quantile(maps, 0.995)
    normalized_maps = ((maps - map_low) / (map_high - map_low).clamp_min(1e-12))
    normalized_maps = normalized_maps.clamp(0, 1)

    config.PIXEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, anomaly_map in zip(names, normalized_maps):
        pixels = (anomaly_map.numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(pixels, mode="L").save(config.PIXEL_OUTPUT_DIR / name)

    image_scores = {name: float(score) for name, score in zip(names, scores)}
    with config.SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(image_scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(names)} images and wrote results to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

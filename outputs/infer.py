from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import config
from anomaly_pipeline import (
    ImageDataset,
    MultiScaleFeatureExtractor,
    extract_features,
    list_images,
    robust_normalize,
    score_feature_map,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = list_images(config.TEST_DIR)
    if not paths:
        raise RuntimeError(f"No test images found in {config.TEST_DIR}")
    if not config.MODEL_PATH.exists():
        raise RuntimeError(f"Run train.py first; missing {config.MODEL_PATH}")

    model = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=config.INFER_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    extractor = MultiScaleFeatureExtractor().to(device)
    raw_maps: dict[str, torch.Tensor] = {}
    raw_scores: dict[str, float] = {}

    for index, (images, names) in enumerate(loader, start=1):
        features = extract_features(extractor, images, device)
        for feature_map, name in zip(features, names):
            pixel_map, image_score = score_feature_map(feature_map, model, device)
            raw_maps[name] = pixel_map
            raw_scores[name] = image_score
        print(f"Scored image {index}/{len(loader)}")

    ordered_names = [path.name for path in paths]
    normalized_image_scores = robust_normalize(
        torch.tensor([raw_scores[name] for name in ordered_names])
    )
    all_pixels = torch.cat([raw_maps[name].flatten() for name in ordered_names])
    pixel_low = torch.quantile(all_pixels, 0.01)
    pixel_high = torch.quantile(all_pixels, 0.99)

    config.PIXEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = {}
    for index, name in enumerate(ordered_names):
        scores[name] = float(normalized_image_scores[index])
        if float(pixel_high - pixel_low) < 1e-12:
            normalized_map = torch.zeros_like(raw_maps[name])
        else:
            normalized_map = (
                (raw_maps[name] - pixel_low) / (pixel_high - pixel_low)
            ).clamp(0.0, 1.0)
        png = np.rint(normalized_map.numpy() * 255.0).astype(np.uint8)
        Image.fromarray(png, mode="L").save(config.PIXEL_OUTPUT_DIR / name)

    with config.SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {len(scores)} image scores and pixel maps to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

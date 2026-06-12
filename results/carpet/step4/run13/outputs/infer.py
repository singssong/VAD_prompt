from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from anomaly import (
    FeatureExtractor,
    ImageDataset,
    extract_features,
    list_images,
    normalize_image_scores,
    normalize_scores,
    score_features,
)
from config import CONFIG


def infer() -> None:
    config = CONFIG
    config.pixel_dir.mkdir(parents=True, exist_ok=True)
    paths = list_images(config.test_dir)
    if not paths:
        raise RuntimeError(f"No test images found in {config.test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config.model_path, map_location="cpu", weights_only=True)
    extractor = FeatureExtractor().to(device)
    projection = checkpoint["projection"].to(device)
    memory_bank = checkpoint["memory_bank"].to(device)
    pixel_low, pixel_high = checkpoint["pixel_range"]
    image_low, image_high = checkpoint["image_range"]

    loader = DataLoader(
        ImageDataset(paths, config.image_size),
        batch_size=config.inference_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    image_scores: dict[str, float] = {}
    for images, names in loader:
        patches, grid_size = extract_features(
            extractor, images.to(device, non_blocking=True), projection
        )
        maps, raw_scores = score_features(patches, grid_size, memory_bank, config)
        maps = normalize_scores(maps, pixel_low, pixel_high)
        maps = F.interpolate(
            maps[:, None],
            size=(config.image_size, config.image_size),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        scores = normalize_image_scores(raw_scores, image_low, image_high)

        for name, anomaly_map, score in zip(names, maps, scores):
            pixels = (
                anomaly_map.mul(255).round().byte().cpu().numpy().astype(np.uint8)
            )
            Image.fromarray(pixels, mode="L").save(config.pixel_dir / name)
            image_scores[name] = float(score.cpu())

    with config.scores_path.open("w", encoding="utf-8") as handle:
        json.dump(image_scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(image_scores)} images and wrote {config.scores_path}")


if __name__ == "__main__":
    infer()

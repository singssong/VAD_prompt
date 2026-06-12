#!/usr/bin/env python3
import json

import torch
from torch.utils.data import DataLoader

import config
from common import (
    ImageDataset,
    MidLevelResNet18,
    aggregate_image_scores,
    bounded_normalize,
    extract_features,
    postprocess_maps,
    save_pixel_map,
    score_feature_maps,
)


def score_images(backbone, loader, model, device):
    image_scores = {}
    config.PIXEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for images, filenames in loader:
        features = extract_features(backbone, images.to(device, non_blocking=True))
        raw_pixel_maps = postprocess_maps(score_feature_maps(features, model))
        raw_image_scores = aggregate_image_scores(raw_pixel_maps)
        normalized_images = bounded_normalize(
            raw_image_scores, model["image_scale"]
        )
        normalized_pixels = bounded_normalize(
            raw_pixel_maps, model["pixel_scale"]
        )

        for index, filename in enumerate(filenames):
            image_scores[filename] = float(normalized_images[index].cpu())
            save_pixel_map(
                normalized_pixels[index, 0],
                config.PIXEL_OUTPUT_DIR / filename,
            )
    return image_scores


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=False)
    dataset = ImageDataset(config.TEST_DIR)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    backbone = MidLevelResNet18().eval().to(device)
    image_scores = score_images(backbone, loader, model, device)

    score_path = config.OUTPUT_DIR / "image_scores.json"
    with score_path.open("w", encoding="utf-8") as handle:
        json.dump(image_scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(image_scores)} images")
    print(f"Saved image scores to {score_path}")
    print(f"Saved pixel maps to {config.PIXEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()


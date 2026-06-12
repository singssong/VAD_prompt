#!/usr/bin/env python3
import json

import torch
from torch.utils.data import DataLoader

import config
from model_utils import (
    FeatureExtractor,
    ImageDataset,
    aggregate_image_scores,
    list_images,
    normalize_image_score,
    save_pixel_map,
    score_feature_maps,
    seed_everything,
)


def score_images(
    extractor: FeatureExtractor,
    loader: DataLoader,
    model: dict,
    device: torch.device,
) -> dict[str, float]:
    scores = {}
    projection = model["projection"].to(device, dtype=torch.float32)
    memory_bank = model["memory_bank"].to(device, dtype=torch.float32)

    with torch.inference_mode():
        for batch_index, (images, names) in enumerate(loader, start=1):
            features = extractor(images.to(device))
            maps = score_feature_maps(features, projection, memory_bank)
            raw_scores = aggregate_image_scores(maps)
            for index, name in enumerate(names):
                scores[name] = normalize_image_score(
                    float(raw_scores[index].item()),
                    model["image_low"],
                    model["image_high"],
                )
                save_pixel_map(
                    maps[index, 0],
                    config.PIXEL_OUTPUT_DIR / name,
                    model["pixel_high"],
                )
            print(f"Scored test batch {batch_index}/{len(loader)}")
    return scores


def main() -> None:
    seed_everything(config.RANDOM_SEED)
    paths = list_images(config.TEST_DIR)
    if not paths:
        raise RuntimeError(f"No test images found in {config.TEST_DIR}")
    if not config.MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: run train.py first ({config.MODEL_PATH})")

    config.PIXEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=config.INFER_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().to(device)
    scores = score_images(extractor, loader, model, device)
    with config.SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved {len(scores)} image scores to {config.SCORES_PATH}")


if __name__ == "__main__":
    main()

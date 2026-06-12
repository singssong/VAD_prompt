import json

import numpy as np
import torch
from PIL import Image

import config
from pipeline import (
    MidLevelFeatureExtractor,
    extract_features,
    make_loader,
    robust_normalize,
    score_feature_maps,
)


def infer() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    extractor = MidLevelFeatureExtractor().eval().to(device)
    projection = checkpoint["projection"].to(device)
    memory_bank = checkpoint["memory_bank"].to(device)
    loader = make_loader(config.TEST_DIR, config.INFER_BATCH_SIZE)

    filenames: list[str] = []
    raw_maps: list[np.ndarray] = []
    raw_scores: list[np.ndarray] = []
    with torch.inference_mode():
        for images, names in loader:
            features = extract_features(
                extractor, images.to(device, non_blocking=True), projection
            )
            pixel_maps, image_scores = score_feature_maps(features, memory_bank)
            filenames.extend(names)
            raw_maps.append(pixel_maps.cpu().numpy())
            raw_scores.append(image_scores.cpu().numpy())

    maps = np.concatenate(raw_maps, axis=0)
    scores = np.concatenate(raw_scores, axis=0)
    normalized_scores = robust_normalize(scores)
    normalized_maps = robust_normalize(maps)

    config.PIXEL_SCORE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, pixel_map in zip(filenames, normalized_maps):
        image = Image.fromarray(np.round(pixel_map * 255.0).astype(np.uint8), mode="L")
        image.save(config.PIXEL_SCORE_DIR / filename)

    score_dict = {
        filename: float(score)
        for filename, score in zip(filenames, normalized_scores)
    }
    with config.IMAGE_SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(score_dict, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(filenames)} images; outputs written to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    infer()


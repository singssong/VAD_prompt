import json

import torch
from torch.utils.data import DataLoader

import config
from common import (
    FeatureExtractor,
    ImageDataset,
    extract_features,
    list_images,
    save_pixel_map,
    score_features,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_paths = list_images(config.TEST_DIR)
    if not test_paths:
        raise RuntimeError(f"No test images found in {config.TEST_DIR}")

    model_data = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    model = FeatureExtractor().eval().to(device)
    loader = DataLoader(
        ImageDataset(test_paths),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    config.PIXEL_DIR.mkdir(parents=True, exist_ok=True)
    scores = {}
    for images, names in loader:
        features = extract_features(model, images.to(device))
        pixel_maps, image_scores = score_features(features, model_data)
        for name, pixel_map, image_score in zip(names, pixel_maps, image_scores):
            save_pixel_map(config.PIXEL_DIR / name, pixel_map)
            scores[name] = float(image_score)

    with config.SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(scores.items())), handle, indent=2)
        handle.write("\n")
    print(f"Wrote scores and maps for {len(scores)} test images")


if __name__ == "__main__":
    main()

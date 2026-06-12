import json

import numpy as np
import torch
from PIL import Image

import config
from pipeline import (
    MidLevelResNet,
    aggregate_image_scores,
    extract_features,
    list_images,
    make_loader,
    normalize_scores,
    resize_maps,
    score_feature_maps,
    seed_everything,
)


def main() -> None:
    seed_everything(config.SEED)
    paths = list_images(config.TEST_DIR)
    if not paths:
        raise RuntimeError(f"No test images found in {config.TEST_DIR}")
    if not config.MODEL_PATH.exists():
        raise RuntimeError(f"Missing model {config.MODEL_PATH}; run train.py first")

    config.PIXEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=False)
    backbone = MidLevelResNet().to(device)
    loader = make_loader(paths)
    scores: dict[str, float] = {}

    for images, filenames in loader:
        features = extract_features(backbone, images, device)
        low_resolution_maps = score_feature_maps(
            features, model["mean"], model["variance"]
        )
        raw_image_scores = aggregate_image_scores(low_resolution_maps)
        image_scores = normalize_scores(
            raw_image_scores, model["image_low"], model["image_high"]
        )
        full_resolution_maps = resize_maps(low_resolution_maps)
        pixel_scores = normalize_scores(
            full_resolution_maps, model["pixel_low"], model["pixel_high"]
        )

        for filename, image_score, pixel_score in zip(
            filenames, image_scores, pixel_scores
        ):
            scores[filename] = float(image_score.cpu())
            output_array = (
                pixel_score.mul(255).round().byte().cpu().numpy().astype(np.uint8)
            )
            Image.fromarray(output_array, mode="L").save(
                config.PIXEL_OUTPUT_DIR / filename
            )

    ordered_scores = {path.name: scores[path.name] for path in paths}
    with config.SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(ordered_scores, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(ordered_scores)} image scores to {config.SCORES_PATH}")
    print(f"Wrote pixel maps to {config.PIXEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

import json

import numpy as np
import torch
from PIL import Image

from config import MODEL_PATH, PIXEL_OUTPUT_DIR, SCORES_PATH, TEST_DIR
from pipeline import (
    build_feature_extractor,
    list_images,
    make_loader,
    score_batch,
    seed_everything,
)


def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = list_images(TEST_DIR)
    if not paths:
        raise RuntimeError(f"No test images found in {TEST_DIR}")

    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    extractor = build_feature_extractor(device)
    loader = make_loader(paths)
    PIXEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = {}

    for images, names in loader:
        images = images.to(device, non_blocking=True)
        image_scores, pixel_maps = score_batch(extractor, images, model)
        for name, score, pixel_map in zip(names, image_scores.cpu(), pixel_maps.cpu()):
            scores[name] = float(score)
            output = np.rint(pixel_map.numpy() * 255.0).astype(np.uint8)
            Image.fromarray(output, mode="L").save(PIXEL_OUTPUT_DIR / name)

    with SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images and wrote results to {SCORES_PATH}.")


if __name__ == "__main__":
    main()

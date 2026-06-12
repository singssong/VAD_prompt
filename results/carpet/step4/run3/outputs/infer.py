from __future__ import annotations

import json

import torch

from anomaly_pipeline import (
    MODEL_PATH,
    OUTPUT_DIR,
    TEST_DIR,
    FeatureExtractor,
    image_files,
    score_images,
    seed_everything,
)


def main() -> None:
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = image_files(TEST_DIR)
    if not paths:
        raise RuntimeError(f"No test images found in {TEST_DIR}")
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Run outputs/train.py first.")

    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    extractor = FeatureExtractor().to(device)
    scores = score_images(extractor, model, paths, device)
    score_path = OUTPUT_DIR / "image_scores.json"
    with score_path.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images; results saved to {score_path}")


if __name__ == "__main__":
    main()

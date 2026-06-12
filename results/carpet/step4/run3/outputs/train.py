from __future__ import annotations

import torch

from anomaly_pipeline import (
    MODEL_PATH,
    TRAIN_DIR,
    FeatureExtractor,
    build_normal_model,
    image_files,
    seed_everything,
)


def main() -> None:
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = image_files(TRAIN_DIR)
    if not paths:
        raise RuntimeError(f"No training images found in {TRAIN_DIR}")

    extractor = FeatureExtractor().to(device)
    model = build_normal_model(extractor, paths, device)
    torch.save(model, MODEL_PATH)
    print(f"Saved normal feature model to {MODEL_PATH}")
    print(f"Training images: {len(paths)}; memory patches: {len(model['memory_bank'])}")


if __name__ == "__main__":
    main()

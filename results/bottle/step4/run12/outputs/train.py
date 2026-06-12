from __future__ import annotations

import torch

import config
from pipeline import (
    FeatureExtractor,
    build_normal_feature_model,
    list_images,
    make_loader,
)


def main() -> None:
    torch.manual_seed(config.RANDOM_SEED)
    train_paths = list_images(config.TRAIN_DIR)
    if not train_paths:
        raise RuntimeError(f"No training images found in {config.TRAIN_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor().to(device)
    model = build_normal_feature_model(
        extractor, make_loader(train_paths, shuffle=False), device
    )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model, config.MODEL_PATH)
    print(
        f"Saved {len(model['memory_bank'])} normal patch features "
        f"from {len(train_paths)} images to {config.MODEL_PATH}"
    )


if __name__ == "__main__":
    main()

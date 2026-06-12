from __future__ import annotations

import torch

import config
from pipeline import FeatureExtractor, build_normal_model, list_images, seed_everything


def main() -> None:
    seed_everything(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_paths = list_images(config.TRAIN_DIR)
    if not train_paths:
        raise RuntimeError(f"No training images found in {config.TRAIN_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor().to(device)
    model = build_normal_model(extractor, train_paths, device)
    torch.save(model, config.MODEL_PATH)
    print(
        f"Saved {len(model['memory_bank'])} normal patch features to "
        f"{config.MODEL_PATH}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import torch

import config
from pipeline import (
    FeatureExtractor,
    build_normal_model,
    fit_score_calibration,
    image_files,
    make_loader,
)


def main() -> None:
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = image_files(config.TRAIN_DIR)
    if not paths:
        raise RuntimeError(f"No training images in {config.TRAIN_DIR}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractor = FeatureExtractor().to(device)
    loader = make_loader(paths)
    model = build_normal_model(extractor, loader, device)
    model["calibration"] = fit_score_calibration(extractor, loader, model, device)
    model["metadata"] = {
        "backbone": config.BACKBONE,
        "layers": config.FEATURE_LAYERS,
        "image_size": config.IMAGE_SIZE,
        "training_images": len(paths),
    }
    torch.save(model, config.MODEL_PATH)
    print(f"Saved model trained on {len(paths)} images to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()

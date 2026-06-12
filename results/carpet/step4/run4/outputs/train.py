import torch

import config
from pipeline import (
    MidLevelResNet,
    collect_training_calibration,
    fit_normal_feature_model,
    list_images,
    make_loader,
    seed_everything,
)


def main() -> None:
    seed_everything(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = list_images(config.TRAIN_DIR)
    if not paths:
        raise RuntimeError(f"No training images found in {config.TRAIN_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = MidLevelResNet().to(device)
    loader = make_loader(paths)

    model = fit_normal_feature_model(backbone, loader, device)
    model.update(collect_training_calibration(backbone, loader, model, device))
    model["config"] = {
        "backbone": config.BACKBONE,
        "feature_layers": config.FEATURE_LAYERS,
        "image_size": config.IMAGE_SIZE,
        "feature_size": config.FEATURE_SIZE,
        "top_fraction": config.TOP_FRACTION,
    }
    torch.save(model, config.MODEL_PATH)
    print(f"Saved normal feature model to {config.MODEL_PATH}")
    print(f"Training images: {len(paths)}; descriptors: {model['sample_count'].item()}")


if __name__ == "__main__":
    main()

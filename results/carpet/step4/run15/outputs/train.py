import torch

import config
from pipeline import (
    MidLevelFeatureExtractor,
    build_normal_feature_model,
    extract_features,
    make_loader,
    make_projection,
)


def train() -> None:
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = MidLevelFeatureExtractor().eval().to(device)
    loader = make_loader(config.TRAIN_DIR, config.TRAIN_BATCH_SIZE)

    projection = make_projection(128 + 256, config.PROJECTION_DIM).to(device)
    feature_batches = []
    for images, _ in loader:
        features = extract_features(
            extractor, images.to(device, non_blocking=True), projection
        )
        feature_batches.append(features.cpu())

    memory_bank = build_normal_feature_model(
        feature_batches, config.MEMORY_BANK_SIZE
    )
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": memory_bank,
            "projection": projection.cpu(),
            "backbone": config.BACKBONE,
            "feature_layers": config.FEATURE_LAYERS,
            "image_size": config.IMAGE_SIZE,
            "patch_grid_size": config.PATCH_GRID_SIZE,
        },
        config.MODEL_PATH,
    )
    print(f"Saved {len(memory_bank)} normal patch features to {config.MODEL_PATH}")


if __name__ == "__main__":
    train()


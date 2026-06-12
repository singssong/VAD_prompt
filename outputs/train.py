from __future__ import annotations

import torch
from torch.utils.data import DataLoader

import config
from anomaly_pipeline import (
    ImageDataset,
    MultiScaleFeatureExtractor,
    extract_features,
    list_images,
    model_normal_features,
)


def main() -> None:
    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = list_images(config.TRAIN_DIR)
    if not paths:
        raise RuntimeError(f"No training images found in {config.TRAIN_DIR}")

    loader = DataLoader(
        ImageDataset(paths),
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    extractor = MultiScaleFeatureExtractor().to(device)
    feature_batches = []
    for batch_index, (images, _) in enumerate(loader, start=1):
        features = extract_features(extractor, images, device)
        feature_batches.append(features.reshape(-1, features.shape[-1]).cpu().half())
        print(f"Extracted batch {batch_index}/{len(loader)}")

    model = model_normal_features(feature_batches, config.MEMORY_BANK_SIZE)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model, config.MODEL_PATH)
    print(
        f"Saved {model['memory_bank'].shape[0]} normal patch features "
        f"to {config.MODEL_PATH}"
    )


if __name__ == "__main__":
    main()

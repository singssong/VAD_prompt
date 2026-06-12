import torch
from torch.utils.data import DataLoader

import config
from common import (
    FeatureExtractor,
    ImageDataset,
    build_normal_model,
    extract_features,
    list_images,
)


def main():
    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = list_images(config.TRAIN_DIR)
    if len(paths) < 2:
        raise RuntimeError(f"Need at least two training images in {config.TRAIN_DIR}")

    split = max(1, min(len(paths) - 1, round(
        len(paths) * (1.0 - config.CALIBRATION_FRACTION)
    )))
    memory_paths, calibration_paths = paths[:split], paths[split:]
    model = FeatureExtractor().eval().to(device)

    def batches(selected_paths):
        loader = DataLoader(
            ImageDataset(selected_paths),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )
        output = []
        for images, _ in loader:
            output.append(extract_features(model, images.to(device)).cpu())
        return output

    memory_features = batches(memory_paths)
    calibration_features = batches(calibration_paths)
    model_data = build_normal_model(memory_features, calibration_features)
    model_data["metadata"] = {
        "backbone": "resnet18_imagenet1k_v1",
        "feature_layers": list(config.FEATURE_LAYERS),
        "image_size": config.IMAGE_SIZE,
        "memory_images": len(memory_paths),
        "calibration_images": len(calibration_paths),
    }
    torch.save(model_data, config.MODEL_PATH)
    print(
        f"Saved {len(model_data['memory_bank'])} normal patches to "
        f"{config.MODEL_PATH}"
    )


if __name__ == "__main__":
    main()

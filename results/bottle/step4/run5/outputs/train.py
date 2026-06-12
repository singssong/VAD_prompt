#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader

import config
from pipeline import (
    ImageDataset,
    MidLevelResNet18,
    aggregate_image_scores,
    create_normal_model,
    extract_features,
    resize_maps,
    score_feature_map,
    smooth_maps,
)


def make_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def feature_batches(model, loader, device):
    for images, _ in loader:
        yield extract_features(model, images.to(device)).cpu()


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset(config.TRAIN_DIR)
    if not dataset.paths:
        raise RuntimeError(f"No training images found in {config.TRAIN_DIR}")

    model = MidLevelResNet18().eval().to(device)
    mean, variance, count = create_normal_model(
        feature_batches(model, make_loader(dataset), device)
    )
    mean_device = mean.to(device)
    variance_device = variance.to(device)

    pixel_values = []
    image_values = []
    for images, _ in make_loader(dataset):
        features = extract_features(model, images.to(device))
        maps = smooth_maps(score_feature_map(features, mean_device, variance_device))
        maps = resize_maps(maps)
        pixel_values.append(maps.flatten().cpu())
        image_values.append(aggregate_image_scores(maps).cpu())

    pixel_values = torch.cat(pixel_values)
    image_values = torch.cat(image_values)
    calibration = {
        "pixel_low": float(
            torch.quantile(pixel_values, config.CALIBRATION_LOW_QUANTILE)
        ),
        "pixel_high": float(
            torch.quantile(pixel_values, config.CALIBRATION_HIGH_QUANTILE)
        ),
        "image_low": float(
            torch.quantile(image_values, config.CALIBRATION_LOW_QUANTILE)
        ),
        "image_high": float(
            torch.quantile(image_values, config.CALIBRATION_HIGH_QUANTILE)
        ),
    }

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean,
            "variance": variance,
            "calibration": calibration,
            "training_images": count,
            "backbone": config.BACKBONE,
            "feature_layers": config.FEATURE_LAYERS,
            "image_size": config.IMAGE_SIZE,
        },
        config.MODEL_PATH,
    )
    print(f"Saved model from {count} normal images to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()


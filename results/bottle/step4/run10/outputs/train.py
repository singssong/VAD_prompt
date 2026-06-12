#!/usr/bin/env python3
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from common import (
    ImageDataset,
    MidLevelResNet18,
    aggregate_image_scores,
    extract_features,
    postprocess_maps,
    quantile,
    score_feature_maps,
)


def make_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def fit_normal_feature_model(backbone, loader, device):
    """Fit a diagonal Gaussian independently at every spatial location."""
    feature_sum = None
    feature_square_sum = None
    sample_count = 0

    for images, _ in loader:
        features = extract_features(backbone, images.to(device, non_blocking=True))
        batch_sum = features.sum(dim=0, keepdim=True)
        batch_square_sum = features.square().sum(dim=0, keepdim=True)
        feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
        feature_square_sum = (
            batch_square_sum
            if feature_square_sum is None
            else feature_square_sum + batch_square_sum
        )
        sample_count += features.shape[0]

    mean = feature_sum / sample_count
    variance = feature_square_sum / sample_count - mean.square()
    variance = variance.clamp_min(config.VARIANCE_EPS)
    return {"mean": mean.cpu(), "variance": variance.cpu()}


def calibrate_scores(backbone, loader, model, device):
    image_scores = []
    sampled_pixels = []
    for images, _ in loader:
        features = extract_features(backbone, images.to(device, non_blocking=True))
        pixel_maps = postprocess_maps(score_feature_maps(features, model))
        image_scores.extend(aggregate_image_scores(pixel_maps).cpu().tolist())
        sampled_pixels.extend(pixel_maps[:, :, ::4, ::4].flatten().cpu().tolist())

    return {
        "image_scale": quantile(image_scores, config.IMAGE_SCALE_QUANTILE),
        "pixel_scale": quantile(sampled_pixels, config.PIXEL_SCALE_QUANTILE),
    }


def main():
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset(config.TRAIN_DIR)
    loader = make_loader(dataset)
    backbone = MidLevelResNet18().eval().to(device)

    model = fit_normal_feature_model(backbone, loader, device)
    model.update(calibrate_scores(backbone, loader, model, device))
    model["metadata"] = {
        "backbone": "ImageNet-pretrained ResNet-18",
        "layers": config.FEATURE_LAYERS,
        "image_size": config.IMAGE_SIZE,
        "feature_size": config.FEATURE_SIZE,
        "training_images": len(dataset),
    }

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model, config.MODEL_PATH)
    print(f"Saved model to {config.MODEL_PATH}")
    print(model["metadata"])
    print(
        f"Calibration: image_scale={model['image_scale']:.6f}, "
        f"pixel_scale={model['pixel_scale']:.6f}"
    )


if __name__ == "__main__":
    main()


import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from pipeline import (
    FeatureExtractor,
    ImageDataset,
    OnlineSpatialGaussian,
    aggregate_image_scores,
    choose_channels,
    extract_features,
    list_images,
    postprocess_maps,
    score_feature_maps,
)


def make_loader(files):
    return DataLoader(
        ImageDataset(files),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def fit_normal_model(extractor, loader, device):
    statistics = OnlineSpatialGaussian()
    channel_indices = None
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        if channel_indices is None:
            with torch.inference_mode():
                all_features = extractor(images)
            channel_indices = choose_channels(all_features.shape[1])
            statistics.update(all_features[:, channel_indices.to(device)])
        else:
            statistics.update(extract_features(extractor, images, channel_indices))
    mean, variance = statistics.finalize()
    return channel_indices, mean, variance


@torch.inference_mode()
def calibrate_normal_scores(extractor, loader, channel_indices, mean, variance, device):
    mean = mean.to(device)
    variance = variance.to(device)
    image_scores = []
    pixel_samples = []
    for images, _ in loader:
        features = extract_features(
            extractor, images.to(device, non_blocking=True), channel_indices
        )
        maps = postprocess_maps(score_feature_maps(features, mean, variance))
        image_scores.extend(aggregate_image_scores(maps).cpu().numpy().tolist())
        pixel_samples.append(maps[:, ::4, ::4].cpu().numpy().reshape(-1))
    pixels = np.concatenate(pixel_samples)
    image_low = float(np.median(image_scores))
    image_high = float(np.quantile(image_scores, config.CALIBRATION_HIGH_QUANTILE))
    pixel_low = float(np.median(pixels))
    pixel_high = float(np.quantile(pixels, config.CALIBRATION_HIGH_QUANTILE))
    return image_low, image_high, pixel_low, pixel_high


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default=str(config.TRAIN_DIR))
    parser.add_argument("--model-path", type=str, default=str(config.MODEL_PATH))
    args = parser.parse_args()

    files = list_images(Path(args.train_dir))
    if not files:
        raise RuntimeError(f"No training images found in {args.train_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor().to(device)
    loader = make_loader(files)
    channel_indices, mean, variance = fit_normal_model(extractor, loader, device)
    calibration = calibrate_normal_scores(
        extractor, loader, channel_indices, mean, variance, device
    )

    model = {
        "backbone": config.BACKBONE,
        "feature_layers": config.FEATURE_LAYERS,
        "image_size": config.IMAGE_SIZE,
        "channel_indices": channel_indices,
        "mean": mean,
        "variance": variance,
        "image_score_low": calibration[0],
        "image_score_high": calibration[1],
        "pixel_score_low": calibration[2],
        "pixel_score_high": calibration[3],
        "training_images": len(files),
    }
    torch.save(model, args.model_path)
    print(f"Saved normal model from {len(files)} images to {args.model_path}")


if __name__ == "__main__":
    main()

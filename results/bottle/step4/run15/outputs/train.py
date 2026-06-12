import torch

import config
from common import (
    FeatureExtractor,
    build_memory_bank,
    extract_feature_batches,
    list_images,
    score_feature_maps,
    seed_everything,
)


def split_normal_images(paths):
    generator = torch.Generator().manual_seed(config.SEED)
    order = torch.randperm(len(paths), generator=generator).tolist()
    calibration_count = max(1, round(len(paths) * config.CALIBRATION_FRACTION))
    calibration_indices = set(order[:calibration_count])
    fit = [path for index, path in enumerate(paths) if index not in calibration_indices]
    calibration = [path for index, path in enumerate(paths) if index in calibration_indices]
    return fit, calibration


def calibrate_scores(extractor, paths, memory_bank, device):
    pixel_values = []
    image_values = []
    for features, _ in extract_feature_batches(
        extractor, paths, device, config.INFER_BATCH_SIZE
    ):
        pixel_maps, image_scores = score_feature_maps(features, memory_bank, device)
        pixel_values.append(pixel_maps.flatten())
        image_values.append(image_scores)
    pixels = torch.cat(pixel_values)
    images = torch.cat(image_values)
    q = config.NORMALIZATION_QUANTILE
    return {
        "pixel_low": float(torch.quantile(pixels, 0.50)),
        "pixel_high": float(torch.quantile(pixels, q)),
        "image_low": float(torch.quantile(images, 0.50)),
        "image_high": float(torch.quantile(images, q)),
    }


def train_normal_model():
    seed_everything(config.SEED)
    paths = list_images(config.TRAIN_DIR)
    if len(paths) < 2:
        raise RuntimeError(f"Need at least two training images in {config.TRAIN_DIR}")

    fit_paths, calibration_paths = split_normal_images(paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor().to(device).eval()

    print(f"Extracting normal features from {len(fit_paths)} images on {device}...")
    memory_bank = build_memory_bank(extractor, fit_paths, device)
    print(f"Built memory bank with {len(memory_bank)} patch descriptors.")

    print(f"Calibrating on {len(calibration_paths)} held-out normal images...")
    calibration = calibrate_scores(extractor, calibration_paths, memory_bank, device)
    artifact = {
        "method": "PatchCore-style nearest-neighbor patch memory bank",
        "backbone": config.BACKBONE,
        "feature_layers": config.FEATURE_LAYERS,
        "image_size": config.IMAGE_SIZE,
        "memory_bank": memory_bank.to(torch.float16),
        "calibration": calibration,
    }
    torch.save(artifact, config.MODEL_PATH)
    print(f"Saved model to {config.MODEL_PATH}")
    print(f"Calibration: {calibration}")


if __name__ == "__main__":
    train_normal_model()

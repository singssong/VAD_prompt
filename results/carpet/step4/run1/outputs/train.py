import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur

import config
from model import (
    ImageDataset,
    ResNetFeatureExtractor,
    extract_features,
    list_images,
    make_projection,
)


def make_loader(paths, shuffle=False):
    return DataLoader(
        ImageDataset(paths),
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.NUM_WORKERS > 0,
    )


@torch.inference_mode()
def build_normal_feature_model(extractor, loader, projection, device):
    """Store a representative memory bank of normal patch features."""
    features = []
    for images, _ in loader:
        batch_features = extract_features(
            extractor, images.to(device, non_blocking=True), projection
        )
        features.append(batch_features.cpu())
    features = torch.cat(features)
    generator = torch.Generator().manual_seed(config.SEED)
    count = min(config.MEMORY_BANK_SIZE, len(features))
    indices = torch.randperm(len(features), generator=generator)[:count]
    return features[indices].contiguous()


@torch.inference_mode()
def nearest_neighbor_distances(features, memory_bank):
    chunks = []
    memory_sq = (memory_bank * memory_bank).sum(dim=1).unsqueeze(0)
    for query in features.split(config.DISTANCE_QUERY_CHUNK):
        query_sq = (query * query).sum(dim=1, keepdim=True)
        distances_sq = query_sq + memory_sq - 2.0 * (query @ memory_bank.T)
        chunks.append(distances_sq.clamp_min_(0).min(dim=1).values.sqrt_())
    return torch.cat(chunks)


def postprocess_maps(flat_scores, batch_size):
    maps = flat_scores.reshape(
        batch_size, 1, config.FEATURE_GRID_SIZE, config.FEATURE_GRID_SIZE
    )
    return gaussian_blur(
        maps,
        kernel_size=[config.GAUSSIAN_KERNEL_SIZE] * 2,
        sigma=[config.GAUSSIAN_SIGMA] * 2,
    )


def aggregate_image_scores(maps):
    flat = maps.flatten(1)
    top_count = max(1, int(flat.shape[1] * config.IMAGE_TOP_FRACTION))
    return flat.topk(top_count, dim=1).values.mean(dim=1)


@torch.inference_mode()
def calibrate_scores(extractor, loader, projection, memory_bank, device):
    """Fit robust pixel and image score ranges on held-out normal images."""
    pixel_scores = []
    image_scores = []
    memory_bank = memory_bank.to(device)
    for images, _ in loader:
        features = extract_features(
            extractor, images.to(device, non_blocking=True), projection
        )
        distances = nearest_neighbor_distances(features, memory_bank)
        maps = postprocess_maps(distances, images.shape[0])
        pixel_scores.append(maps.flatten().cpu())
        image_scores.append(aggregate_image_scores(maps).cpu())

    pixels = torch.cat(pixel_scores).float()
    images = torch.cat(image_scores).float()
    pixel_low, pixel_high = torch.quantile(
        pixels, torch.tensor([0.50, 0.999])
    ).tolist()
    image_low, image_high = torch.quantile(
        images, torch.tensor([0.05, 0.995])
    ).tolist()
    eps = 1e-6
    return {
        "pixel_low": float(pixel_low),
        "pixel_high": float(max(pixel_high, pixel_low + eps)),
        "image_low": float(image_low),
        "image_high": float(max(image_high, image_low + eps)),
    }


def split_training_images(paths):
    paths = list(paths)
    random.Random(config.SEED).shuffle(paths)
    calibration_count = max(1, round(len(paths) * config.CALIBRATION_FRACTION))
    return paths[calibration_count:], paths[:calibration_count]


def main():
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = list_images(config.TRAIN_DIR)
    if len(paths) < 2:
        raise RuntimeError(f"Need at least two training images in {config.TRAIN_DIR}")

    reference_paths, calibration_paths = split_training_images(paths)
    extractor = ResNetFeatureExtractor().to(device)
    projection = make_projection(device)

    memory_bank = build_normal_feature_model(
        extractor, make_loader(reference_paths), projection, device
    )
    calibration = calibrate_scores(
        extractor,
        make_loader(calibration_paths),
        projection,
        memory_bank,
        device,
    )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": memory_bank,
            "projection": projection.cpu(),
            "calibration": calibration,
            "backbone": "resnet18",
            "feature_layers": config.FEATURE_LAYERS,
            "image_size": config.IMAGE_SIZE,
            "feature_grid_size": config.FEATURE_GRID_SIZE,
            "training_image_count": len(paths),
        },
        config.MODEL_PATH,
    )
    print(
        f"Saved {len(memory_bank)} normal patch features from "
        f"{len(reference_paths)} images to {config.MODEL_PATH}"
    )
    print(f"Calibrated on {len(calibration_paths)} held-out normal images")


if __name__ == "__main__":
    main()

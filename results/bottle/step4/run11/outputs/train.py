#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from model_utils import (
    FeatureExtractor,
    ImageDataset,
    aggregate_image_scores,
    build_normal_model,
    flatten_and_project,
    list_images,
    make_projection,
    nearest_neighbor_distances,
    seed_everything,
)


def extract_training_features(
    extractor: FeatureExtractor, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, int]:
    batches = []
    channels = 0
    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(loader, start=1):
            features = extractor(images.to(device))
            channels = features.shape[1]
            batches.append(features.cpu())
            print(f"Extracted training batch {batch_index}/{len(loader)}")
    return torch.cat(batches), channels


def model_normal_features(
    feature_maps: torch.Tensor, projection: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    all_patches = []
    for features in feature_maps.split(config.TRAIN_BATCH_SIZE):
        all_patches.append(flatten_and_project(features, projection).cpu())
    all_patches = torch.cat(all_patches)

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    count = min(config.MEMORY_BANK_SIZE, len(all_patches))
    indices = torch.randperm(len(all_patches), generator=generator)[:count]
    memory_bank = all_patches[indices].contiguous()
    return all_patches, memory_bank


def calibrate_normal_scores(
    all_patches: torch.Tensor,
    memory_bank: torch.Tensor,
    image_count: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    patches_per_image = all_patches.shape[0] // image_count
    image_scores = []
    sampled_pixels = []
    memory_bank = memory_bank.to(device)
    for image_index in range(image_count):
        patches = all_patches[
            image_index * patches_per_image : (image_index + 1) * patches_per_image
        ].to(device)
        distances = nearest_neighbor_distances(
            patches, memory_bank, config.NN_QUERY_CHUNK
        )
        score = aggregate_image_scores(distances.reshape(1, 1, 32, 32))
        image_scores.append(float(score.item()))
        sampled_pixels.append(distances.cpu().numpy())
    return np.asarray(image_scores), np.concatenate(sampled_pixels)


def main() -> None:
    seed_everything(config.RANDOM_SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = list_images(config.TRAIN_DIR)
    if not paths:
        raise RuntimeError(f"No training images found in {config.TRAIN_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().to(device)
    feature_maps, channel_count = extract_training_features(extractor, loader, device)
    projection = make_projection(
        channel_count, config.PROJECTION_DIM, config.RANDOM_SEED
    )
    all_patches, memory_bank = model_normal_features(feature_maps, projection)
    image_scores, pixel_scores = calibrate_normal_scores(
        all_patches, memory_bank, len(paths), device
    )
    model = build_normal_model(
        all_patches, projection, image_scores, pixel_scores
    )
    torch.save(model, config.MODEL_PATH)
    print(
        f"Saved {len(model['memory_bank'])} normal patch features to "
        f"{config.MODEL_PATH}"
    )


if __name__ == "__main__":
    main()

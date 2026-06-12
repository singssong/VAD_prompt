import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import ImageDataset, MidLevelResNet18, extract_features
from config import (
    BATCH_SIZE,
    FEATURE_LAYERS,
    IMAGE_SIZE,
    MODEL_PATH,
    NUM_WORKERS,
    PATCHES_PER_TRAIN_IMAGE,
    RANDOM_SEED,
    TRAIN_DIR,
)


def build_normal_feature_model(model, loader, device):
    """Estimate channel statistics and retain a representative normal patch bank."""
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    feature_sum = None
    feature_sq_sum = None
    feature_count = 0
    sampled_patches = []

    for images, _ in loader:
        features = extract_features(model, images.to(device))
        flat = features.reshape(-1, features.shape[-1])
        batch_sum = flat.double().sum(dim=0).cpu()
        batch_sq_sum = flat.double().square().sum(dim=0).cpu()
        feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
        feature_sq_sum = (
            batch_sq_sum if feature_sq_sum is None else feature_sq_sum + batch_sq_sum
        )
        feature_count += flat.shape[0]

        patch_count = features.shape[1] * features.shape[2]
        sample_count = min(PATCHES_PER_TRAIN_IMAGE, patch_count)
        for image_features in features:
            indices = torch.randperm(patch_count, generator=generator)[:sample_count]
            sampled_patches.append(
                image_features.reshape(patch_count, -1)[indices.to(device)].cpu()
            )

    mean = (feature_sum / feature_count).float()
    variance = (feature_sq_sum / feature_count - mean.double().square()).clamp_min(1e-8)
    std = variance.sqrt().float()
    memory_bank = torch.cat(sampled_patches, dim=0)
    memory_bank = torch.nn.functional.normalize((memory_bank - mean) / std, dim=1)
    return {
        "memory_bank": memory_bank.half(),
        "feature_mean": mean,
        "feature_std": std,
    }


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset(TRAIN_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    model = MidLevelResNet18().eval().to(device)
    artifact = build_normal_feature_model(model, loader, device)
    artifact["metadata"] = {
        "method": "PatchCore-style nearest-neighbor normal feature bank",
        "backbone": "ImageNet-pretrained ResNet-18",
        "feature_layers": FEATURE_LAYERS,
        "image_size": IMAGE_SIZE,
        "train_image_count": len(dataset),
    }
    torch.save(artifact, MODEL_PATH)
    print(f"Saved {len(artifact['memory_bank'])} normal patches to {MODEL_PATH}")


if __name__ == "__main__":
    main()

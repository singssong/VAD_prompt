import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import v2

import config


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.transform = v2.Compose(
            [
                v2.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            return self.transform(image), self.paths[index].name


def list_images(directory):
    return sorted(
        path
        for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def build_feature_extractor(device):
    backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
    extractor = create_feature_extractor(
        backbone, return_nodes={layer: layer for layer in config.FEATURE_LAYERS}
    )
    return extractor.eval().to(device)


@torch.inference_mode()
def extract_features(extractor, images, projection):
    outputs = extractor(images)
    target_size = outputs[config.FEATURE_LAYERS[0]].shape[-2:]
    levels = []
    for layer in config.FEATURE_LAYERS:
        feature = F.avg_pool2d(outputs[layer], kernel_size=3, stride=1, padding=1)
        if feature.shape[-2:] != target_size:
            feature = F.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=False
            )
        levels.append(feature)
    features = torch.cat(levels, dim=1)
    features = features.permute(0, 2, 3, 1)
    features = features @ projection
    return F.normalize(features, dim=-1)


def create_projection(extractor, device):
    dummy = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    with torch.inference_mode():
        outputs = extractor(dummy)
    input_dim = sum(outputs[layer].shape[1] for layer in config.FEATURE_LAYERS)
    generator = torch.Generator(device=device).manual_seed(config.SEED)
    projection = torch.randn(
        input_dim, config.EMBEDDING_DIM, generator=generator, device=device
    )
    return projection / np.sqrt(config.EMBEDDING_DIM)


@torch.inference_mode()
def collect_feature_patches(extractor, paths, projection, device):
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    batches = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        features = extract_features(extractor, images, projection)
        batches.append(features.reshape(-1, config.EMBEDDING_DIM).cpu())
    return torch.cat(batches)


def build_normal_model(feature_patches):
    generator = torch.Generator().manual_seed(config.SEED)
    count = min(config.MEMORY_BANK_SIZE, len(feature_patches))
    indices = torch.randperm(len(feature_patches), generator=generator)[:count]
    return feature_patches[indices].contiguous()


def nearest_neighbor_distances(features, memory_bank):
    result = []
    memory_bank = memory_bank.to(features.device)
    for chunk in features.split(config.KNN_CHUNK_SIZE):
        # Unit-normalized embeddings make 1-cosine similarity an anomaly distance.
        similarities = chunk @ memory_bank.T
        result.append(1.0 - similarities.max(dim=1).values)
    return torch.cat(result)


@torch.inference_mode()
def score_normal_images(extractor, paths, projection, memory_bank, device):
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    image_scores = []
    patch_scores = []
    memory_bank = memory_bank.to(device)
    for images, _ in loader:
        features = extract_features(
            extractor, images.to(device, non_blocking=True), projection
        )
        distances = nearest_neighbor_distances(
            features.reshape(-1, config.EMBEDDING_DIM), memory_bank
        )
        top_count = max(1, round(distances.numel() * config.TOP_FRACTION))
        image_scores.append(distances.topk(top_count).values.mean().item())
        patch_scores.append(distances.cpu())
    return np.asarray(image_scores), torch.cat(patch_scores).numpy()


def fit_calibration(image_scores, patch_scores):
    image_center = float(np.median(image_scores))
    image_scale = float(np.quantile(image_scores, 0.95) - image_center)
    patch_low = float(np.quantile(patch_scores, 0.50))
    patch_high = float(np.quantile(patch_scores, 0.995))
    return {
        "image_center": image_center,
        "image_scale": max(image_scale, 1e-6),
        "patch_low": patch_low,
        "patch_high": max(patch_high, patch_low + 1e-6),
    }


def main():
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = list_images(config.TRAIN_DIR)
    if not paths:
        raise RuntimeError(f"No training images found in {config.TRAIN_DIR}")

    shuffled = paths.copy()
    random.Random(config.SEED).shuffle(shuffled)
    calibration_count = max(1, round(len(shuffled) * config.CALIBRATION_FRACTION))
    calibration_paths = shuffled[:calibration_count]
    memory_paths = shuffled[calibration_count:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = build_feature_extractor(device)
    projection = create_projection(extractor, device)
    patches = collect_feature_patches(extractor, memory_paths, projection, device)
    memory_bank = build_normal_model(patches)
    image_scores, patch_scores = score_normal_images(
        extractor, calibration_paths, projection, memory_bank, device
    )
    calibration = fit_calibration(image_scores, patch_scores)

    artifact = {
        "backbone": config.BACKBONE,
        "feature_layers": config.FEATURE_LAYERS,
        "image_size": config.IMAGE_SIZE,
        "embedding_dim": config.EMBEDDING_DIM,
        "projection": projection.cpu(),
        "memory_bank": memory_bank.half(),
        "calibration": calibration,
    }
    torch.save(artifact, config.MODEL_PATH)
    summary = {
        "training_images": len(paths),
        "memory_images": len(memory_paths),
        "calibration_images": len(calibration_paths),
        "memory_patches": len(memory_bank),
        "calibration": calibration,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a one-class PatchCore-style anomaly detector on normal images."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat((layer2, layer3), dim=1)
        return F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)


def list_images(root: Path):
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


@torch.inference_mode()
def extract_all(model, loader, device):
    batches = []
    for images in loader:
        features = model(images.to(device, non_blocking=True))
        batches.append(features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu())
    return torch.cat(batches)


def project(features, projection, mean, scale):
    features = (features - mean) / scale
    features = features @ projection
    return F.normalize(features, dim=1)


def nearest_distances(queries, memory, device, query_chunk=2048, memory_chunk=4096):
    memory = memory.to(device)
    results = []
    for start in range(0, len(queries), query_chunk):
        query = queries[start : start + query_chunk].to(device)
        best = torch.full((len(query),), -1.0, device=device)
        for memory_start in range(0, len(memory), memory_chunk):
            similarity = query @ memory[memory_start : memory_start + memory_chunk].T
            best = torch.maximum(best, similarity.max(dim=1).values)
        results.append((1.0 - best).clamp_min_(0).cpu())
    return torch.cat(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("./data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("./outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--memory-size", type=int, default=12000)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = FeatureExtractor().eval().to(device)

    print(f"Extracting patches from {len(dataset)} normal images on {device}...")
    raw_features = extract_all(model, loader, device)
    feature_mean = raw_features.mean(dim=0)
    feature_scale = raw_features.std(dim=0).clamp_min(1e-4)

    generator = torch.Generator().manual_seed(args.seed)
    projection = torch.randn(
        raw_features.shape[1], args.projection_dim, generator=generator
    ) / np.sqrt(args.projection_dim)
    projected = project(raw_features, projection, feature_mean, feature_scale)

    memory_size = min(args.memory_size, len(projected))
    indices = torch.randperm(len(projected), generator=generator)[:memory_size]
    memory = projected[indices].contiguous()

    # Estimate a fixed heatmap scale exclusively from held-out normal patches.
    calibration_count = min(30000, len(projected))
    calibration_indices = torch.randperm(len(projected), generator=generator)[:calibration_count]
    calibration = nearest_distances(projected[calibration_indices], memory, device)
    positive = calibration[calibration > 1e-7]
    if len(positive) < 100:
        positive = calibration
    score_low = float(torch.quantile(positive, 0.90))
    score_high = float(torch.quantile(positive, 0.999))
    if score_high <= score_low:
        score_high = score_low + 1e-6

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "PatchCore-style nearest-neighbor patch memory",
            "backbone": "wide_resnet50_2 IMAGENET1K_V2",
            "projection": projection,
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "memory": memory.to(torch.float16),
            "score_low": score_low,
            "score_high": score_high,
            "image_size": 256,
            "feature_grid": 32,
            "seed": args.seed,
        },
        args.model_out,
    )
    print(f"Saved {memory_size} normal patch embeddings to {args.model_out}")


if __name__ == "__main__":
    main()

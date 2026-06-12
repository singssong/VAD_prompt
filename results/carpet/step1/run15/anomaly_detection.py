#!/usr/bin/env python3
"""Unsupervised image anomaly scoring using a compact PatchCore memory bank."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.paths = sorted(
            p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.paths[index].name


class PatchFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, images):
        x = self.layer1(self.stem(images))
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        features = torch.cat((layer2, layer3), dim=1)
        # Local averaging makes descriptors less sensitive to one-pixel texture shifts.
        return F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)


def make_loader(root, transform, batch_size, workers, shuffle=False):
    dataset = ImageDataset(root, transform)
    if not dataset.paths:
        raise RuntimeError(f"No images found in {root}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


@torch.inference_mode()
def collect_training_patches(model, loader, device):
    chunks = []
    for images, _ in loader:
        features = model(images.to(device, non_blocking=True))
        patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        chunks.append(patches.cpu())
    return torch.cat(chunks)


def fit_projection(patches, output_dim, seed):
    # Standardize channels using normal data before a Johnson-Lindenstrauss projection.
    mean = patches.mean(dim=0)
    std = patches.std(dim=0).clamp_min(1e-6)
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(
        patches.shape[1], output_dim, generator=generator, dtype=torch.float32
    ) / np.sqrt(output_dim)
    return mean, std, projection


def project_patches(patches, mean, std, projection):
    return F.normalize(((patches - mean) / std) @ projection, dim=1)


def build_memory_bank(patches, mean, std, projection, memory_size, seed):
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(patches.shape[0], generator=generator)[
        : min(memory_size, patches.shape[0])
    ]
    return project_patches(patches[selected], mean, std, projection).contiguous()


def nearest_distances(query, memory, chunk_size=2048):
    # Vectors are unit-normalized, so squared Euclidean distance is 2 - 2*cosine.
    results = []
    memory_t = memory.T.contiguous()
    for chunk in query.split(chunk_size):
        similarity = chunk @ memory_t
        results.append((2.0 - 2.0 * similarity.max(dim=1).values).clamp_min_(0).sqrt_())
    return torch.cat(results)


@torch.inference_mode()
def score_images(model, loader, device, mean, std, projection, memory):
    mean = mean.to(device)
    std = std.to(device)
    projection = projection.to(device)
    memory = memory.to(device)
    rows = []

    for images, names in loader:
        features = model(images.to(device, non_blocking=True))
        batch, channels, height, width = features.shape
        patches = features.permute(0, 2, 3, 1).reshape(batch, height * width, channels)

        for index, name in enumerate(names):
            query = project_patches(patches[index], mean, std, projection)
            distances = nearest_distances(query, memory)
            # Averaging the worst 1% is stable while retaining sensitivity to small defects.
            top_k = max(1, int(round(distances.numel() * 0.01)))
            score = distances.topk(top_k).values.mean().item()
            rows.append((name, score))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-size", type=int, default=40000)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    transform = weights.transforms(crop_size=256, resize_size=256)
    train_loader = make_loader(
        args.train_dir, transform, args.batch_size, args.workers
    )
    test_loader = make_loader(args.test_dir, transform, args.batch_size, args.workers)

    model = PatchFeatureExtractor().eval().to(device)
    train_patches = collect_training_patches(model, train_loader, device)
    mean, std, projection = fit_projection(
        train_patches, args.projection_dim, args.seed
    )
    memory = build_memory_bank(
        train_patches, mean, std, projection, args.memory_size, args.seed
    )
    del train_patches

    rows = score_images(model, test_loader, device, mean, std, projection, memory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "anomaly_score"])
        writer.writerows((name, f"{score:.8f}") for name, score in rows)

    print(f"Wrote {len(rows)} scores to {args.output}")
    print("Method: PatchCore-style nearest-neighbor patch memory")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2 (layer2 + layer3)")


if __name__ == "__main__":
    main()

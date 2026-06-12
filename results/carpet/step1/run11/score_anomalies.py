#!/usr/bin/env python3
"""Score test images with an unsupervised PatchCore-style memory bank."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import v2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, directory: Path, transform):
        self.paths = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {directory}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.paths[index].name


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self._features = {}

        self.layer2.register_forward_hook(
            lambda _module, _inputs, output: self._features.__setitem__("layer2", output)
        )
        self.layer3.register_forward_hook(
            lambda _module, _inputs, output: self._features.__setitem__("layer3", output)
        )

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, images):
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        _ = self.layer3(x)

        layer2 = local_average(self._features["layer2"])
        layer3 = local_average(self._features["layer3"])
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat((layer2, layer3), dim=1)


def local_average(features):
    return F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)


def flatten_patches(features):
    return features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])


@torch.inference_mode()
def collect_train_patches(model, loader, device):
    batches = []
    for images, _names in loader:
        features = model(images.to(device, non_blocking=True))
        batches.append(flatten_patches(features).cpu())
    return torch.cat(batches)


def build_memory_bank(patches, size, seed):
    """Build a reproducible representative subset of the reference patches."""
    generator = torch.Generator().manual_seed(seed)
    patches = patches.float()
    if len(patches) <= size:
        return patches
    indices = torch.randperm(len(patches), generator=generator)[:size]
    return patches[indices].contiguous()


@torch.inference_mode()
def score_images(model, loader, memory_bank, device):
    memory_bank = memory_bank.to(device)
    names = []
    raw_scores = []

    for images, batch_names in loader:
        features = model(images.to(device, non_blocking=True))
        batch_size, _channels, height, width = features.shape
        patches = features.permute(0, 2, 3, 1).reshape(batch_size, height * width, -1)

        for image_patches, name in zip(patches, batch_names):
            nearest = torch.full(
                (len(image_patches),), float("inf"), device=device
            )
            for memory_chunk in memory_bank.split(2048):
                distances = torch.cdist(image_patches, memory_chunk)
                nearest = torch.minimum(nearest, distances.min(dim=1).values)

            # Defects cover multiple nearby patches; a high-tail mean is more stable
            # than a single maximum while retaining sensitivity to small anomalies.
            tail_size = max(1, round(0.01 * nearest.numel()))
            score = nearest.topk(tail_size).values.mean()
            names.append(name)
            raw_scores.append(score.item())

    return names, np.asarray(raw_scores, dtype=np.float64)


def robust_normalize(scores):
    low, high = np.percentile(scores, [5, 95])
    if high <= low:
        low, high = scores.min(), scores.max()
    if high <= low:
        return np.zeros_like(scores)
    return np.clip((scores - low) / (high - low), 0.0, 1.0)


def write_scores(output_path, names, raw_scores):
    normalized = robust_normalize(raw_scores)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "anomaly_score", "raw_score"])
        for name, score, raw_score in zip(names, normalized, raw_scores):
            writer.writerow([name, f"{score:.10f}", f"{raw_score:.10f}"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--memory-size", type=int, default=12_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = v2.Compose(
        [
            v2.Resize((256, 256), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    train_loader = DataLoader(
        ImageDataset(args.train_dir, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ImageDataset(args.test_dir, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    print(f"Extracting reference patches on {device}...")
    model = FeatureExtractor().to(device)
    train_patches = collect_train_patches(model, train_loader, device)
    print(f"Building memory bank from {len(train_patches):,} patches...")
    memory_bank = build_memory_bank(train_patches, args.memory_size, args.seed)
    del train_patches

    print(f"Scoring {len(test_loader.dataset)} test images...")
    names, raw_scores = score_images(model, test_loader, memory_bank, device)
    write_scores(args.output, names, raw_scores)
    print(f"Wrote {len(names)} scores to {args.output}")
    print("Method: PatchCore-style nearest-neighbor patch anomaly detection")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2 (layers 2 and 3)")


if __name__ == "__main__":
    main()

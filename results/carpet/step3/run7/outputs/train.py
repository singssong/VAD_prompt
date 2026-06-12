#!/usr/bin/env python3
"""Train a PatchCore-style one-class anomaly detector."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_paths(directory: Path):
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


class ImageDataset(Dataset):
    def __init__(self, directory: Path):
        self.paths = image_paths(directory)
        if not self.paths:
            raise RuntimeError(f"No images found in {directory}")
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std, path.name


class FeatureExtractor(nn.Module):
    """Wide-ResNet feature extractor returning layers 1, 2 and 3."""

    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return f1, f2, f3


def make_projection(input_dim=1792, output_dim=256, seed=1337):
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    projection /= np.sqrt(output_dim)
    return projection


def patch_embeddings(features, projection):
    target_size = features[1].shape[-2:]
    resized = []
    for feature in features:
        feature = F.adaptive_avg_pool2d(feature, target_size) if feature.shape[-2:] != target_size else feature
        resized.append(F.normalize(feature, dim=1))
    merged = torch.cat(resized, dim=1).permute(0, 2, 3, 1)
    embedded = merged @ projection
    return F.normalize(embedded, dim=-1)


@torch.no_grad()
def second_neighbor_scale(bank, device, sample_count=5000, chunk_size=512):
    count = min(sample_count, len(bank))
    generator = torch.Generator().manual_seed(2026)
    indices = torch.randperm(len(bank), generator=generator)[:count]
    queries = bank[indices].to(device)
    reference = bank.to(device)
    distances = []
    for start in range(0, count, chunk_size):
        similarity = queries[start : start + chunk_size] @ reference.T
        nearest_two = torch.topk(similarity, k=2, dim=1).values
        distances.append((1.0 - nearest_two[:, 1]).cpu())
    values = torch.cat(distances)
    return max(float(torch.quantile(values, 0.995)), 1e-4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patches-per-image", type=int, default=144)
    parser.add_argument("--max-bank-size", type=int, default=40000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = FeatureExtractor().to(device)
    projection = make_projection().to(device)
    sampled_batches = []

    for batch_index, (images, _) in enumerate(loader):
        embeddings = patch_embeddings(model(images.to(device)), projection)
        flat = embeddings.reshape(embeddings.shape[0], -1, embeddings.shape[-1])
        for image_patches in flat:
            choice = torch.randperm(image_patches.shape[0], device=device)[: args.patches_per_image]
            sampled_batches.append(image_patches[choice].cpu())
        print(f"\rExtracting normal features: {min((batch_index + 1) * args.batch_size, len(dataset))}/{len(dataset)}", end="")
    print()

    bank = torch.cat(sampled_batches)
    if len(bank) > args.max_bank_size:
        bank = bank[torch.randperm(len(bank))[: args.max_bank_size]]
    bank = F.normalize(bank.float(), dim=1).contiguous()
    calibration_scale = second_neighbor_scale(bank, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": bank.half(),
            "projection": projection.cpu(),
            "calibration_scale": calibration_scale,
            "backbone": "wide_resnet50_2_imagenet1k_v2",
            "image_size": 256,
        },
        args.output,
    )
    print(f"Saved {len(bank)} normal patch embeddings to {args.output}")
    print(f"Pixel-map calibration scale: {calibration_scale:.6f}")


if __name__ == "__main__":
    main()

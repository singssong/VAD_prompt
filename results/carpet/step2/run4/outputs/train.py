#!/usr/bin/env python3
"""Build a PatchCore-style normal patch memory bank."""

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


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms(
            crop_size=256, resize_size=256
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.backbone.eval()

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(x)
        layer3 = self.backbone.layer3(layer2)
        layer2 = F.avg_pool2d(layer2, 3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, 3, stride=1, padding=1)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((layer2, layer3), dim=1)


def list_images(directory):
    return sorted(
        path for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_features(model, paths, device, batch_size):
    loader = DataLoader(
        ImageDataset(paths), batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda"
    )
    all_features = []
    with torch.inference_mode():
        for images in loader:
            features = model(images.to(device, non_blocking=True))
            features = features.permute(0, 2, 3, 1).flatten(0, 2).cpu()
            all_features.append(features)
    return torch.cat(all_features)


def nearest_distances(queries, bank, device, chunk_size=2048):
    bank = bank.to(device)
    bank_norm = (bank * bank).sum(dim=1)
    result = []
    for start in range(0, len(queries), chunk_size):
        query = queries[start:start + chunk_size].to(device)
        distances = (
            (query * query).sum(dim=1, keepdim=True)
            + bank_norm.unsqueeze(0)
            - 2.0 * query @ bank.T
        )
        result.append(distances.clamp_min_(0).min(dim=1).values.sqrt().cpu())
    return torch.cat(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="./data/train")
    parser.add_argument("--model-out", default="./outputs/model.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bank-size", type=int, default=12000)
    parser.add_argument("--projection-dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    paths = list_images(args.train_dir)
    if not paths:
        raise RuntimeError(f"No training images found in {args.train_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor().to(device)

    # Reserve normal images solely to establish a stable visualization scale.
    shuffled = paths.copy()
    random.shuffle(shuffled)
    calibration_count = max(1, round(0.1 * len(shuffled)))
    calibration_paths = shuffled[:calibration_count]
    bank_paths = shuffled[calibration_count:]

    raw_bank = collect_features(model, bank_paths, device, args.batch_size)
    generator = torch.Generator().manual_seed(args.seed)
    projection = torch.randn(
        raw_bank.shape[1], args.projection_dim, generator=generator
    ) / np.sqrt(args.projection_dim)
    projected = raw_bank @ projection

    # The texture is spatially stationary; random patch sampling preserves its
    # normal appearance distribution while keeping exact NN inference tractable.
    count = min(args.bank_size, len(projected))
    indices = torch.randperm(len(projected), generator=generator)[:count]
    memory_bank = projected[indices].contiguous()

    calibration_raw = collect_features(model, calibration_paths, device, args.batch_size)
    calibration = calibration_raw @ projection
    calibration_distances = nearest_distances(calibration, memory_bank, device)
    calibration_distances = calibration_distances.reshape(len(calibration_paths), 32, 32)
    position_center = torch.quantile(calibration_distances, 0.50, dim=0)
    position_high = torch.quantile(calibration_distances, 0.99, dim=0)
    position_scale = (position_high - position_center).clamp_min(0.25)

    output = Path(args.model_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": memory_bank,
            "projection": projection,
            "position_center": position_center,
            "position_scale": position_scale,
            "map_low": 0.0,
            "map_high": 2.0,
            "feature_grid": 32,
            "seed": args.seed,
            "backbone": "wide_resnet50_2_imagenet1k_v2",
        },
        output,
    )
    print(
        f"Saved {len(memory_bank)} normal patches from {len(bank_paths)} images "
        f"to {output} with position-wise normal calibration"
    )


if __name__ == "__main__":
    main()

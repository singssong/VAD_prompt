#!/usr/bin/env python3
"""Build a PatchCore-style normal feature memory bank."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, pil_to_tensor, resize


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            image = resize(image, [256, 256], InterpolationMode.BILINEAR, antialias=True)
            tensor = pil_to_tensor(image).float().div_(255.0)
        return normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, images):
        features = self.layer1(self.stem(images))
        layer2 = self.layer2(features)
        layer3 = self.layer3(layer2)
        # Both levels are represented on the 16x16 layer3 patch grid.
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=2, padding=1)
        return torch.cat([layer2, layer3], dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-size", type=int, default=30000)
    parser.add_argument("--calibration-size", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def nearest_distances(queries, memory, chunk_size=1024):
    results = []
    memory_t = memory.T.contiguous()
    for chunk in queries.split(chunk_size):
        similarity = chunk @ memory_t
        results.append(1.0 - similarity.max(dim=1).values)
    return torch.cat(results)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().eval().to(device)

    all_features = []
    with torch.inference_mode():
        for images in loader:
            features = extractor(images.to(device, non_blocking=True))
            features = F.normalize(features, dim=1)
            features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
            all_features.append(features.cpu())
    all_features = torch.cat(all_features)

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(all_features), generator=generator)
    memory_count = min(args.memory_size, len(all_features))
    memory = all_features[permutation[:memory_count]].contiguous()

    # Estimate a robust normal-distance range using held-out normal patches.
    remaining = permutation[memory_count:]
    if len(remaining) == 0:
        remaining = permutation[: min(args.calibration_size, len(permutation))]
        calibration_memory = memory[1:]
    else:
        remaining = remaining[: args.calibration_size]
        calibration_memory = memory
    calibration = all_features[remaining].to(device)
    calibration_memory = calibration_memory.to(device)
    with torch.inference_mode():
        normal_distances = nearest_distances(calibration, calibration_memory).cpu()
    map_low = float(torch.quantile(normal_distances, 0.01))
    map_high = float(torch.quantile(normal_distances, 0.995))
    if map_high <= map_low:
        map_high = map_low + 1e-6

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": memory.half(),
            "map_low": map_low,
            "map_high": map_high,
            "backbone": "wide_resnet50_2",
            "image_size": 256,
            "feature_grid": 16,
            "seed": args.seed,
        },
        args.model_out,
    )
    print(
        f"Saved {memory_count} normal patch features from {len(dataset)} images "
        f"to {args.model_out} (device={device}, calibration={map_low:.6f}..{map_high:.6f})"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a normal patch-feature memory bank from training images."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.transform = ResNet50_Weights.IMAGENET1K_V2.transforms(
            crop_size=256, resize_size=256
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            return self.transform(image)


class PatchFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool, model.layer1
        )
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        layer1 = self.stem(images)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        features = torch.cat((layer2, layer3), dim=1)
        features = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        return features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])


def make_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    return projection / np.sqrt(output_dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-size", type=int, default=30000)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    paths = image_files(args.train_dir)
    if not paths:
        raise RuntimeError(f"No training images found in {args.train_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = PatchFeatureExtractor().eval().to(device)
    projection = make_projection(1536, args.projection_dim, args.seed).to(device)
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    feature_batches = []
    with torch.inference_mode():
        for images in loader:
            features = extractor(images.to(device, non_blocking=True))
            features = F.normalize(features @ projection, dim=1)
            feature_batches.append(features.cpu())

    all_features = torch.cat(feature_batches)
    generator = torch.Generator().manual_seed(args.seed)
    if len(all_features) > args.memory_size:
        indices = torch.randperm(len(all_features), generator=generator)[:args.memory_size]
        memory_bank = all_features[indices]
    else:
        memory_bank = all_features

    # Training nearest-neighbor distances provide a robust visualization scale.
    scale_sample = memory_bank[: min(4000, len(memory_bank))].to(device)
    bank_device = memory_bank.to(device)
    nearest = []
    with torch.inference_mode():
        for chunk in scale_sample.split(256):
            similarity = chunk @ bank_device.T
            values = similarity.topk(k=min(2, len(bank_device)), dim=1).values
            neighbor_similarity = values[:, -1] if values.shape[1] == 2 else values[:, 0]
            nearest.append((1.0 - neighbor_similarity).cpu())
    normal_distances = torch.cat(nearest)
    map_scale = max(float(torch.quantile(normal_distances, 0.995)) * 4.0, 1e-6)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": memory_bank.half(),
            "projection": projection.cpu(),
            "map_scale": map_scale,
            "backbone": "torchvision resnet50 IMAGENET1K_V2",
            "input_size": 256,
            "seed": args.seed,
        },
        args.model_out,
    )
    print(
        f"Saved {len(memory_bank)} normal patch features from {len(paths)} images "
        f"to {args.model_out}"
    )


if __name__ == "__main__":
    main()

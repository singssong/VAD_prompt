#!/usr/bin/env python3
"""Train a compact PatchCore-style one-class anomaly detector."""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.transform = Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms(
            crop_size=256, resize_size=256
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class FeatureExtractor(nn.Module):
    """Return locally aggregated layer2 and layer3 feature maps."""

    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.layer1(self.stem(images))
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat((layer2, layer3), dim=1)
        return F.normalize(features, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--memory-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = image_files(args.train_dir)
    if not paths:
        raise RuntimeError(f"No training images found in {args.train_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        ImageDataset(paths), batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == "cuda"
    )
    extractor = FeatureExtractor().eval().to(device)

    banks = []
    patches_per_image = math.ceil(args.memory_size / len(paths))
    generator = torch.Generator().manual_seed(args.seed)
    channel_indices = None
    with torch.inference_mode():
        for images in loader:
            features = extractor(images.to(device, non_blocking=True))
            if channel_indices is None:
                channel_count = features.shape[1]
                channel_indices = torch.randperm(channel_count, generator=generator)[:256].sort().values
            features = F.normalize(features[:, channel_indices.to(device)], dim=1)
            patch_grid = features.permute(0, 2, 3, 1).reshape(
                features.shape[0], -1, features.shape[1]
            ).cpu()
            for image_patches in patch_grid:
                selected = torch.randperm(len(image_patches), generator=generator)[:patches_per_image]
                banks.append(image_patches[selected])

    all_patches = torch.cat(banks)
    count = min(args.memory_size, len(all_patches))
    selected = torch.randperm(len(all_patches), generator=generator)[:count]
    memory_bank = all_patches[selected].contiguous().to(torch.float16)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "PatchCore-style random patch memory bank",
            "backbone": "wide_resnet50_2",
            "memory_bank": memory_bank,
            "channel_indices": channel_indices,
            "feature_grid": list(features.shape[-2:]),
            "train_image_count": len(paths),
            "seed": args.seed,
        },
        args.model_path,
    )
    print(
        f"Saved {count} normal patch embeddings from {len(paths)} images "
        f"to {args.model_path} using {device}."
    )


if __name__ == "__main__":
    main()

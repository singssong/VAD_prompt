#!/usr/bin/env python3
"""Build a PatchCore-style memory bank from normal training images only."""

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


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.files = image_files(root)
        if not self.files:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB"))


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, images):
        features = self.layer1(self.stem(images))
        layer2 = self.layer2(features)
        layer3 = self.layer3(layer2)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        # Local averaging makes descriptors less sensitive to one-pixel shifts.
        layer2 = F.avg_pool2d(layer2, 3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, 3, stride=1, padding=1)
        return torch.cat((layer2, layer3), dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bank-size", type=int, default=30000)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    dataset = ImageDataset(args.train_dir, weights.transforms())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, len(dataset)),
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().eval().to(device)

    generator = torch.Generator().manual_seed(args.seed)
    projection = torch.randn(1536, args.embedding_dim, generator=generator)
    projection /= torch.sqrt(torch.tensor(float(args.embedding_dim)))
    projection = projection.to(device)

    descriptors = []
    with torch.inference_mode():
        for images in loader:
            features = extractor(images.to(device, non_blocking=True))
            patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
            patches = F.normalize(patches @ projection, dim=1)
            descriptors.append(patches.cpu())

    descriptors = torch.cat(descriptors)
    count = min(args.bank_size, len(descriptors))
    indices = torch.randperm(len(descriptors), generator=generator)[:count]
    memory_bank = descriptors[indices].to(torch.float16)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "PatchCore-style random memory bank",
            "backbone": "ImageNet Wide ResNet-50-2",
            "projection": projection.cpu(),
            "memory_bank": memory_bank,
            "input_size": 224,
            "feature_grid": 28,
            "seed": args.seed,
            "train_images": len(dataset),
        },
        args.model_out,
    )
    print(
        f"Saved {count:,} normal patch descriptors from {len(dataset)} images "
        f"to {args.model_out} using {device}."
    )


if __name__ == "__main__":
    main()

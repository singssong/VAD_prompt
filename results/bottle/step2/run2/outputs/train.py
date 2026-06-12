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
from torchvision.models import ResNet18_Weights, resnet18

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(root: Path):
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.files = image_files(root)
        if not self.files:
            raise RuntimeError(f"No images found in {root}")
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms(
            crop_size=256, resize_size=256, antialias=True
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB")), self.files[index].name


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        x = self.layer1(self.stem(x))
        raw_f2 = self.layer2(x)
        raw_f3 = self.layer3(raw_f2)
        f2 = F.avg_pool2d(raw_f2, 3, 1, 1)
        f3 = F.avg_pool2d(raw_f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat((f2, f3), dim=1)
        return F.normalize(features, dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--memory-size", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=args.device.startswith("cuda")
    )
    extractor = FeatureExtractor().to(args.device)
    patches = []
    with torch.inference_mode():
        for images, _ in loader:
            features = extractor(images.to(args.device, non_blocking=True))
            patches.append(features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu())

    patches = torch.cat(patches)
    generator = torch.Generator().manual_seed(args.seed)
    keep = min(args.memory_size, len(patches))
    indices = torch.randperm(len(patches), generator=generator)[:keep]
    memory = patches[indices].to(torch.float16).contiguous()

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory": memory,
            "backbone": "resnet18_imagenet1k_v1",
            "feature_layers": ["layer2", "layer3"],
            "input_size": 256,
            "seed": args.seed,
            "training_images": len(dataset),
        },
        args.model_path,
    )
    print(f"Saved {keep} normal patches from {len(dataset)} images to {args.model_path}")


if __name__ == "__main__":
    main()

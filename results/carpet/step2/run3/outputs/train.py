#!/usr/bin/env python3
"""Build a normal patch-feature memory bank from training images."""

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


class ImageDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3

    def forward(self, images):
        x = self.layer1(self.stem(images))
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        layer2 = F.normalize(layer2, dim=1)
        layer3 = F.normalize(layer3, dim=1)
        return torch.cat((layer2, layer3), dim=1)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=script_dir.parent / "data" / "train")
    parser.add_argument("--model-path", type=Path, default=script_dir / "patchcore_model.pt")
    parser.add_argument("--bank-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    weights = ResNet18_Weights.IMAGENET1K_V1
    dataset = ImageDataset(args.train_dir, weights.transforms())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, len(dataset)),
        pin_memory=args.device.startswith("cuda"),
    )
    extractor = FeatureExtractor().to(args.device).eval()
    feature_batches = []
    with torch.inference_mode():
        for images in loader:
            features = extractor(images.to(args.device, non_blocking=True))
            patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
            feature_batches.append(F.normalize(patches, dim=1).cpu())

    all_patches = torch.cat(feature_batches)
    generator = torch.Generator().manual_seed(args.seed)
    if len(all_patches) > args.bank_size:
        indices = torch.randperm(len(all_patches), generator=generator)[: args.bank_size]
        memory_bank = all_patches[indices].contiguous()
    else:
        memory_bank = all_patches.contiguous()

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": memory_bank,
            "backbone": "resnet18",
            "weights": "IMAGENET1K_V1",
            "feature_layers": ["layer2", "layer3"],
            "feature_grid": list(features.shape[-2:]),
            "train_images": len(dataset),
            "seed": args.seed,
        },
        args.model_path,
    )
    print(f"Saved {len(memory_bank):,} normal patches from {len(dataset)} images to {args.model_path}")


if __name__ == "__main__":
    main()

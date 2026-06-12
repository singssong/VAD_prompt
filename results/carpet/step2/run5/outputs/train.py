#!/usr/bin/env python3
"""Build a PatchCore-style normal-patch memory bank."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.files = sorted(
            p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
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
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        x = self.stem(images)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        layer2 = F.normalize(F.avg_pool2d(layer2, 3, 1, 1), dim=1)
        layer3 = F.normalize(F.avg_pool2d(layer3, 3, 1, 1), dim=1)
        return torch.cat((layer2, layer3), dim=1)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=384)
    parser.add_argument("--memory-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    dataset = ImageDataset(args.train_dir, weights.transforms())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().to(device).eval()

    raw_dim = 512 + 1024
    generator = torch.Generator(device=device).manual_seed(args.seed)
    projection = torch.randn(
        raw_dim, args.projection_dim, generator=generator, device=device
    ) / np.sqrt(args.projection_dim)

    patch_batches = []
    with torch.inference_mode():
        for batch_index, images in enumerate(loader, 1):
            features = extractor(images.to(device, non_blocking=True))
            patches = features.permute(0, 2, 3, 1).reshape(-1, raw_dim)
            patches = F.normalize(patches @ projection, dim=1)
            patch_batches.append(patches.cpu())
            print(f"Extracted batch {batch_index}/{len(loader)}", flush=True)

    all_patches = torch.cat(patch_batches)
    keep = min(args.memory_size, len(all_patches))
    selection = torch.randperm(len(all_patches), generator=torch.Generator().manual_seed(args.seed))
    memory_bank = all_patches[selection[:keep]].contiguous()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": memory_bank,
            "projection": projection.cpu(),
            "backbone": "wide_resnet50_2",
            "weights": "IMAGENET1K_V2",
            "feature_layers": ["layer2", "layer3"],
            "input_size": 224,
            "seed": args.seed,
        },
        args.output,
    )
    print(
        f"Saved {keep} normal patch descriptors from {len(dataset)} images to {args.output}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a normal patch-feature memory bank for one-class anomaly detection."""

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
from torchvision.transforms import functional as TF


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.files = sorted(
            p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.files:
            raise RuntimeError(f"No supported images found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image, [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True
            )
            tensor = TF.to_tensor(image)
        tensor = TF.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return tensor


class PatchFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        network = wide_resnet50_2(weights=weights)
        self.stem = torch.nn.Sequential(
            network.conv1, network.bn1, network.relu, network.maxpool
        )
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3

    def forward(self, images):
        x = self.stem(images)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat([layer2, layer3], dim=1)


def make_projection(input_dim: int, output_dim: int, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    projection /= np.sqrt(output_dim)
    return projection


def projected_patches(features, projection):
    patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    patches = patches @ projection
    return F.normalize(patches, dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bank-size", type=int, default=40000)
    parser.add_argument("--projection-dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


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
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    extractor = PatchFeatureExtractor().eval().to(device)
    projection = make_projection(1536, args.projection_dim, args.seed).to(device)

    all_patches = []
    with torch.inference_mode():
        for batch_index, images in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            patches = projected_patches(extractor(images), projection)
            all_patches.append(patches.cpu())
            print(f"\rExtracting train features: {batch_index}/{len(loader)}", end="", flush=True)
    print()

    all_patches = torch.cat(all_patches)
    generator = torch.Generator().manual_seed(args.seed)
    if len(all_patches) > args.bank_size:
        indices = torch.randperm(len(all_patches), generator=generator)[: args.bank_size]
        memory_bank = all_patches[indices].contiguous()
    else:
        memory_bank = all_patches.contiguous()

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": memory_bank,
            "projection": projection.cpu(),
            "backbone": "wide_resnet50_2",
            "weights": "IMAGENET1K_V2",
            "image_size": 256,
            "feature_grid": 32,
            "seed": args.seed,
        },
        args.model_out,
    )
    print(
        f"Saved {len(memory_bank)} normal patch embeddings from "
        f"{len(dataset)} images to {args.model_out}"
    )


if __name__ == "__main__":
    main()

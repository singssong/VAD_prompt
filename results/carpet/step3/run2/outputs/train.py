#!/usr/bin/env python3
"""Train a feature-memory anomaly detector using only normal images."""

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


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image, [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True
            )
            tensor = TF.to_tensor(image)
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class FeatureExtractor(torch.nn.Module):
    """Return fused layer2/layer3 patch descriptors on a 32x32 grid."""

    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

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


def project_features(features, projection):
    patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    patches = patches @ projection
    return F.normalize(patches, dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("./data/train"))
    parser.add_argument("--model-path", type=Path, default=Path("./outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--memory-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, (len(dataset) + args.batch_size - 1) // args.batch_size),
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().eval().to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    projection = torch.randn(
        1536, args.projection_dim, generator=generator, device=device
    ) / np.sqrt(args.projection_dim)

    all_patches = []
    with torch.inference_mode():
        for batch_index, images in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            features = extractor(images)
            patches = project_features(features, projection)
            all_patches.append(patches.cpu())
            print(f"Extracted batch {batch_index}/{len(loader)}", flush=True)

    all_patches = torch.cat(all_patches)
    memory_size = min(args.memory_size, len(all_patches))
    sample_generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(all_patches), generator=sample_generator)[:memory_size]
    memory_bank = all_patches[indices].contiguous()

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "PatchCore-style nearest-neighbor feature memory bank",
            "backbone": "wide_resnet50_2",
            "image_size": 256,
            "feature_grid": 32,
            "projection": projection.cpu(),
            "memory_bank": memory_bank,
            "seed": args.seed,
        },
        args.model_path,
    )
    print(
        f"Saved {memory_size} normal patch features from {len(dataset)} images "
        f"to {args.model_path}"
    )


if __name__ == "__main__":
    main()

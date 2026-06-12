#!/usr/bin/env python3
"""Train a one-class PatchCore-style anomaly detector on normal images."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import Compose, Normalize, ToTensor


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            return self.transform(image)


def make_backbone(device: torch.device):
    model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
    model = create_feature_extractor(
        model, return_nodes={"layer2": "layer2", "layer3": "layer3"}
    )
    return model.eval().to(device)


def patch_features(backbone, images, projection):
    features = backbone(images)
    layer2 = F.avg_pool2d(features["layer2"], 3, stride=1, padding=1)
    layer3 = F.avg_pool2d(features["layer3"], 3, stride=1, padding=1)
    layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear",
                           align_corners=False)
    combined = torch.cat((layer2, layer3), dim=1)
    patches = combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1])
    return patches @ projection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-size", type=int, default=60000)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=device.type == "cuda")
    backbone = make_backbone(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    projection = torch.randn(
        1536, args.projection_dim, generator=generator, device=device
    ) / np.sqrt(args.projection_dim)

    all_patches = []
    with torch.inference_mode():
        for batch_index, images in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            patches = patch_features(backbone, images, projection)
            all_patches.append(patches.cpu())
            print(f"\rExtracting train features: {batch_index}/{len(loader)}",
                  end="", flush=True)
    print()

    patches = torch.cat(all_patches)
    if len(patches) > args.memory_size:
        selection = torch.randperm(len(patches), generator=torch.Generator().manual_seed(
            args.seed
        ))[:args.memory_size]
        patches = patches[selection]

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "memory_bank": patches.float(),
        "projection": projection.cpu().float(),
        "image_size": 256,
        "grid_size": 32,
        "backbone": "wide_resnet50_2_imagenet1k_v2",
        "method": "PatchCore-style random coreset nearest-neighbor",
        "seed": args.seed,
    }, args.model_out)
    print(f"Saved {len(patches):,} normal patches to {args.model_out}")


if __name__ == "__main__":
    main()

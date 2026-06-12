#!/usr/bin/env python3
"""Train a spatially aware PatchCore-style one-class anomaly model."""

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, files: list[Path]):
        self.files = files
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB"))


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool, net.layer1
        )
        self.layer2 = net.layer2
        self.layer3 = net.layer3

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        # Local averaging reduces sensitivity to harmless pixel-level noise.
        x2 = F.avg_pool2d(x2, kernel_size=3, stride=1, padding=1)
        x3 = F.avg_pool2d(x3, kernel_size=3, stride=1, padding=1)
        x3 = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat((x2, x3), dim=1)
        return F.normalize(features, dim=1)


def add_coordinates(features: torch.Tensor, weight: float) -> torch.Tensor:
    batch, _, height, width = features.shape
    ys = torch.linspace(-1, 1, height, device=features.device)
    xs = torch.linspace(-1, 1, width, device=features.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack((xx, yy)).unsqueeze(0).expand(batch, -1, -1, -1)
    return torch.cat((features, weight * coords), dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--memory-size", type=int, default=30000)
    parser.add_argument("--coordinate-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    files = image_files(args.train_dir)
    if not files:
        raise RuntimeError(f"No training images found in {args.train_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor().eval().to(device)
    loader = DataLoader(
        ImageDataset(files), batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == "cuda",
    )

    descriptors = []
    with torch.inference_mode():
        for images in loader:
            features = add_coordinates(
                model(images.to(device, non_blocking=True)), args.coordinate_weight
            )
            descriptors.append(features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu())

    bank = torch.cat(descriptors)
    generator = torch.Generator().manual_seed(args.seed)
    if len(bank) > args.memory_size:
        bank = bank[torch.randperm(len(bank), generator=generator)[:args.memory_size]]
    bank = bank.to(torch.float16).contiguous()

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "memory_bank": bank,
        "coordinate_weight": args.coordinate_weight,
        "image_size": 256,
        "feature_grid": 32,
        "backbone": "resnet18_imagenet",
        "method": "spatial_patchcore",
        "seed": args.seed,
    }, args.model_path)
    print(f"Saved {len(bank):,} normal patch descriptors to {args.model_path}")


if __name__ == "__main__":
    main()

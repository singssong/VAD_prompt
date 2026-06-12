#!/usr/bin/env python3
"""Train a PaDiM anomaly detector using only normal images."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.files = image_files(root)
        if not self.files:
            raise RuntimeError(f"No images found in {root}")
        self.transform = Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms(
            crop_size=256, resize_size=256
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB"))


class FeatureExtractor(nn.Module):
    """Wide ResNet feature pyramid used by PaDiM."""

    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        size = (32, 32)
        return torch.cat(
            [
                F.adaptive_avg_pool2d(f1, size),
                f2,
                F.interpolate(f3, size=size, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )


@torch.inference_mode()
def extract_embeddings(loader, model, channel_indices, device):
    batches = []
    for images in loader:
        features = model(images.to(device, non_blocking=True))
        batches.append(features[:, channel_indices].cpu())
    return torch.cat(batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    model = FeatureExtractor().to(device)

    total_channels = 256 + 512 + 1024
    generator = torch.Generator().manual_seed(args.seed)
    channel_indices = torch.randperm(total_channels, generator=generator)[: args.feature_dim]
    embeddings = extract_embeddings(loader, model, channel_indices.to(device), device)
    n, d, h, w = embeddings.shape
    flat = embeddings.flatten(2)
    mean = flat.mean(dim=0)
    centered = flat - mean.unsqueeze(0)

    # One regularized covariance matrix is learned at each spatial location.
    covariance = torch.einsum("ndl,nel->lde", centered, centered) / max(n - 1, 1)
    covariance += 0.01 * torch.eye(d).unsqueeze(0)
    precision = torch.linalg.inv(covariance)

    diff = flat - mean.unsqueeze(0)
    train_scores = torch.einsum("ndl,lde,nel->nl", diff, precision, diff)
    train_scores = train_scores.clamp_min_(0).sqrt_()
    calibration_low = torch.quantile(train_scores, 0.98).item()
    calibration_high = torch.quantile(train_scores, 0.9995).item()
    if calibration_high <= calibration_low:
        calibration_high = calibration_low + 1.0

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "PaDiM",
            "backbone": "wide_resnet50_2_imagenet1k_v2",
            "channel_indices": channel_indices,
            "mean": mean,
            "precision": precision,
            "feature_size": (h, w),
            "calibration_low": calibration_low,
            "calibration_high": calibration_high,
            "train_image_count": len(dataset),
            "seed": args.seed,
        },
        args.model_out,
    )
    print(f"Trained PaDiM on {len(dataset)} images; checkpoint: {args.model_out}")


if __name__ == "__main__":
    main()

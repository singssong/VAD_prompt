#!/usr/bin/env python3
"""Fit a PaDiM-style one-class anomaly detector on normal images."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import v2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, directory: Path):
        self.paths = image_files(directory)
        if not self.paths:
            raise RuntimeError(f"No images found in {directory}")
        self.transform = v2.Compose(
            [
                v2.Resize((256, 256), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        self.stem = torch.nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(self.stem(images))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        features = [
            F.adaptive_avg_pool2d(x1, (32, 32)),
            F.interpolate(x2, size=(32, 32), mode="bilinear", align_corners=False),
            F.interpolate(x3, size=(32, 32), mode="bilinear", align_corners=False),
        ]
        return torch.cat(features, dim=1)


@torch.inference_mode()
def collect_features(
    extractor: FeatureExtractor,
    loader: DataLoader,
    channel_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batches = []
    for images in loader:
        features = extractor(images.to(device, non_blocking=True))
        batches.append(features[:, channel_indices].cpu())
    return torch.cat(batches, dim=0)


def anomaly_maps(
    features: torch.Tensor, mean: torch.Tensor, precision: torch.Tensor
) -> torch.Tensor:
    # B,C,H,W -> B,P,C, with a separate normal distribution at every position.
    batch, channels, height, width = features.shape
    vectors = features.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    delta = vectors - mean.unsqueeze(0)
    squared = torch.einsum("bpc,pcd,bpd->bp", delta, precision, delta)
    maps = squared.clamp_min(0).sqrt().reshape(batch, 1, height, width)
    maps = F.interpolate(maps, size=(256, 256), mode="bilinear", align_corners=False)
    return maps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
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
        num_workers=min(4, (len(dataset) + args.batch_size - 1) // args.batch_size),
        pin_memory=device.type == "cuda",
    )

    extractor = FeatureExtractor().eval().to(device)
    total_channels = 256 + 512 + 1024
    generator = torch.Generator().manual_seed(args.seed)
    channel_indices = torch.randperm(total_channels, generator=generator)[
        : args.feature_dim
    ].sort().values
    features = collect_features(extractor, loader, channel_indices.to(device), device)

    # Estimate a regularized full covariance independently at each spatial cell.
    vectors = features.permute(2, 3, 0, 1).reshape(32 * 32, len(dataset), args.feature_dim)
    vectors = vectors.to(device)
    mean = vectors.mean(dim=1)
    centered = vectors - mean.unsqueeze(1)
    covariance = torch.einsum("pnc,pnd->pcd", centered, centered)
    covariance /= max(len(dataset) - 1, 1)
    diagonal_scale = covariance.diagonal(dim1=-2, dim2=-1).mean(dim=1)
    regularization = (0.01 * diagonal_scale).clamp_min(1e-4)
    identity = torch.eye(args.feature_dim, device=device).unsqueeze(0)
    covariance += regularization[:, None, None] * identity
    precision = torch.linalg.inv(covariance)

    # Derive a fixed visualization scale using normal training data only.
    normal_maps = []
    mean_cpu = mean.cpu()
    precision_cpu = precision.cpu()
    for batch in features.split(args.batch_size):
        normal_maps.append(anomaly_maps(batch, mean_cpu, precision_cpu))
    normal_values = torch.cat(normal_maps).flatten()
    pixel_scale = float(torch.quantile(normal_values, 0.999).clamp_min(1e-6))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "PaDiM spatial Gaussian features",
            "backbone": "torchvision wide_resnet50_2 ImageNet-1K",
            "channel_indices": channel_indices,
            "mean": mean_cpu,
            "precision": precision_cpu,
            "pixel_scale": pixel_scale,
            "image_size": 256,
            "map_size": 32,
            "seed": args.seed,
        },
        args.output,
    )
    print(
        f"Trained on {len(dataset)} normal images; saved model to {args.output} "
        f"(device={device}, feature_dim={args.feature_dim})."
    )


if __name__ == "__main__":
    main()

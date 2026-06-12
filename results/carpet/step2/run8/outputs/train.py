#!/usr/bin/env python3
"""Fit a one-class multiscale Gaussian feature model on normal images."""

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
FEATURE_DIM = 128
SAMPLES_PER_IMAGE = 512


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms(
            crop_size=256, resize_size=256, antialias=True
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        x = self.stem(images)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        size = f1.shape[-2:]
        return torch.cat(
            [
                f1,
                F.interpolate(f2, size=size, mode="bilinear", align_corners=False),
                F.interpolate(f3, size=size, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
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
        num_workers=min(4, (len(dataset) + args.batch_size - 1) // args.batch_size),
        pin_memory=device.type == "cuda",
    )

    extractor = FeatureExtractor().eval().to(device)
    generator = torch.Generator().manual_seed(args.seed)
    all_channels = 64 + 128 + 256
    channel_indices = torch.randperm(all_channels, generator=generator)[:FEATURE_DIM]
    sampled_features = []

    with torch.inference_mode():
        for images in loader:
            features = extractor(images.to(device, non_blocking=True))
            features = features[:, channel_indices.to(device)]
            features = features.permute(0, 2, 3, 1).reshape(images.shape[0], -1, FEATURE_DIM)
            for image_features in features:
                indices = torch.randperm(
                    image_features.shape[0], generator=generator, device="cpu"
                )[:SAMPLES_PER_IMAGE].to(device)
                sampled_features.append(image_features[indices].cpu())

    samples = torch.cat(sampled_features).double()
    mean = samples.mean(dim=0)
    centered = samples - mean
    covariance = centered.T @ centered / (samples.shape[0] - 1)
    regularization = 0.01 * covariance.diag().mean()
    covariance += regularization * torch.eye(FEATURE_DIM, dtype=torch.float64)
    precision = torch.linalg.inv(covariance)

    # Calibrate map intensities from normal training patches.
    calibration_scores = torch.einsum(
        "nd,df,nf->n", centered, precision, centered
    ).clamp_min_(0).sqrt_()
    map_low = float(torch.quantile(calibration_scores, 0.90))
    map_high = float(torch.quantile(calibration_scores, 0.9995))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.pt"
    torch.save(
        {
            "channel_indices": channel_indices,
            "mean": mean.float(),
            "precision": precision.float(),
            "map_low": map_low,
            "map_high": map_high,
            "feature_dim": FEATURE_DIM,
            "backbone": "resnet18_imagenet1k_v1",
        },
        model_path,
    )
    print(
        f"Trained on {len(dataset)} normal images; saved {model_path} "
        f"(device={device}, calibration={map_low:.3f}..{map_high:.3f})"
    )


if __name__ == "__main__":
    main()

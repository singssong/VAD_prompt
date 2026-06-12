#!/usr/bin/env python3
"""Fit a PaDiM model using only normal training images."""

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
from torchvision.transforms import v2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.paths = sorted(
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def make_extractor(device):
    backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
    extractor = create_feature_extractor(
        backbone, return_nodes={"layer1": "f1", "layer2": "f2", "layer3": "f3"}
    )
    return extractor.eval().to(device)


def combine_features(outputs, channel_indices):
    target_size = outputs["f2"].shape[-2:]
    features = [
        F.interpolate(outputs["f1"], target_size, mode="bilinear", align_corners=False),
        outputs["f2"],
        F.interpolate(outputs["f3"], target_size, mode="bilinear", align_corners=False),
    ]
    features = torch.cat(features, dim=1)[:, channel_indices]
    return features.flatten(2).transpose(1, 2)


def anomaly_maps(features, mean, cholesky, output_size=256):
    delta = features - mean.unsqueeze(0)
    rhs = delta.permute(1, 2, 0)
    whitened = torch.linalg.solve_triangular(cholesky, rhs, upper=False)
    distances = whitened.square().sum(dim=1).sqrt().transpose(0, 1)
    side = int(distances.shape[1] ** 0.5)
    maps = distances.reshape(-1, 1, side, side)
    maps = F.interpolate(
        maps, size=(output_size, output_size), mode="bilinear", align_corners=False
    )
    return v2.functional.gaussian_blur(maps, kernel_size=[21, 21], sigma=[4.0, 4.0])


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
    transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=Wide_ResNet50_2_Weights.DEFAULT.transforms().mean,
            std=Wide_ResNet50_2_Weights.DEFAULT.transforms().std,
        ),
    ])
    dataset = ImageDataset(args.train_dir, transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == "cuda"
    )
    extractor = make_extractor(device)

    total_channels = 256 + 512 + 1024
    generator = torch.Generator().manual_seed(args.seed)
    channel_indices = torch.randperm(total_channels, generator=generator)[:args.feature_dim]
    channel_indices = channel_indices.sort().values.to(device)

    feature_batches = []
    with torch.inference_mode():
        for images in loader:
            outputs = extractor(images.to(device, non_blocking=True))
            feature_batches.append(combine_features(outputs, channel_indices).cpu())
    features = torch.cat(feature_batches).to(device)
    mean = features.mean(dim=0)
    centered = features - mean.unsqueeze(0)
    covariance = torch.einsum("nld,nle->lde", centered, centered)
    covariance /= max(len(dataset) - 1, 1)
    eye = torch.eye(args.feature_dim, device=device).unsqueeze(0)
    covariance += 0.01 * eye
    cholesky, info = torch.linalg.cholesky_ex(covariance)
    if torch.any(info):
        covariance += 0.05 * eye
        cholesky = torch.linalg.cholesky(covariance)

    train_maps = []
    with torch.inference_mode():
        for batch in features.split(args.batch_size):
            train_maps.append(anomaly_maps(batch, mean, cholesky).cpu())
    train_maps = torch.cat(train_maps)
    flat = train_maps.flatten()
    pixel_low = torch.quantile(flat, 0.95).item()
    pixel_high = torch.quantile(flat, 0.999).item()
    top_k = max(1, int(0.01 * 256 * 256))
    image_scores = train_maps.flatten(1).topk(top_k, dim=1).values.mean(dim=1)
    image_center = image_scores.median().item()
    image_scale = (image_scores - image_center).abs().median().item() * 1.4826
    image_scale = max(image_scale, 1e-6)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "method": "PaDiM",
        "backbone": "wide_resnet50_2",
        "input_size": 224,
        "feature_dim": args.feature_dim,
        "channel_indices": channel_indices.cpu(),
        "mean": mean.cpu(),
        "cholesky": cholesky.cpu(),
        "pixel_low": pixel_low,
        "pixel_high": pixel_high,
        "image_center": image_center,
        "image_scale": image_scale,
    }, args.model_out)
    print(f"Trained PaDiM on {len(dataset)} images; saved {args.model_out}")


if __name__ == "__main__":
    main()

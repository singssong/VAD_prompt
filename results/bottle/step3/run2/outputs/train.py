#!/usr/bin/env python3
"""Train a spatial feature-distribution anomaly detector on normal images."""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_files(root: Path) -> list[Path]:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No supported images found in {root}")
    return files


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image,
                [256, 256],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            tensor = TF.to_tensor(image)
            tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor, path.name


class ResNet18Features(nn.Module):
    """ImageNet ResNet-18 features aligned onto a 32x32 patch grid."""

    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        feature1 = self.layer1(x)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature1 = F.interpolate(feature1, size=(32, 32), mode="bilinear", align_corners=False)
        feature3 = F.interpolate(feature3, size=(32, 32), mode="bilinear", align_corners=False)
        return torch.cat((feature1, feature2, feature3), dim=1)


def gaussian_kernel(size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    kernel1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel1d /= kernel1d.sum()
    return torch.outer(kernel1d, kernel1d).view(1, 1, size, size)


def anomaly_maps(
    features: torch.Tensor,
    mean: torch.Tensor,
    denominator: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    squared_z = (features - mean).square() / denominator
    maps = torch.sqrt(squared_z.mean(dim=1, keepdim=True).clamp_min(0))
    padding = kernel.shape[-1] // 2
    maps = F.pad(maps, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(maps, kernel)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = image_files(args.train_dir)
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    device = torch.device(args.device)
    extractor = ResNet18Features().to(device).eval()

    feature_sum = None
    feature_square_sum = None
    count = 0
    with torch.inference_mode():
        for images, _ in loader:
            features = extractor(images.to(device, non_blocking=True)).double()
            batch_sum = features.sum(dim=0)
            batch_square_sum = features.square().sum(dim=0)
            feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
            feature_square_sum = (
                batch_square_sum
                if feature_square_sum is None
                else feature_square_sum + batch_square_sum
            )
            count += features.shape[0]

    mean = feature_sum / count
    variance = (feature_square_sum / count - mean.square()).clamp_min(0)
    # A small channel-wise global variance floor prevents unstable scores at
    # nearly constant spatial positions while preserving spatial sensitivity.
    global_variance = variance.mean(dim=(1, 2), keepdim=True)
    denominator = variance + 0.01 * global_variance + 1e-6
    mean = mean.float()
    denominator = denominator.float()
    kernel = gaussian_kernel().to(device)

    calibration_chunks = []
    with torch.inference_mode():
        for images, _ in loader:
            features = extractor(images.to(device, non_blocking=True))
            maps = anomaly_maps(features, mean, denominator, kernel)
            calibration_chunks.append(maps.cpu().flatten())
    calibration = torch.cat(calibration_chunks).numpy()
    map_low = float(np.quantile(calibration, 0.50))
    map_high = float(np.quantile(calibration, 0.999))
    if not math.isfinite(map_high) or map_high <= map_low:
        raise RuntimeError("Failed to derive a valid anomaly-map calibration range")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean.cpu(),
            "denominator": denominator.cpu(),
            "map_low": map_low,
            "map_high": map_high,
            "image_size": 256,
            "feature_grid": 32,
            "backbone": "resnet18_imagenet1k_v1",
            "method": "spatial_diagonal_mahalanobis",
            "train_image_count": len(paths),
        },
        args.model_path,
    )
    print(
        f"Saved model to {args.model_path} using {len(paths)} normal images "
        f"(map calibration {map_low:.4f}..{map_high:.4f})"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit a spatial Gaussian model to ImageNet ResNet-18 patch features."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image, [256, 256], interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.to_tensor(image)
        return TF.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )


class FeatureExtractor(torch.nn.Module):
    """Return fused layer2/layer3 feature maps on a 32x32 patch grid."""

    def __init__(self) -> None:
        super().__init__()
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(
            network.conv1, network.bn1, network.relu, network.maxpool
        )
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.layer1(self.stem(images))
        layer2 = self.layer2(features)
        layer3 = self.layer3(layer2)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        fused = torch.cat([layer2, layer3], dim=1)
        return F.normalize(fused, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    paths = image_files(args.train_dir)
    if not paths:
        raise RuntimeError(f"No images found in {args.train_dir}")

    loader = DataLoader(
        ImageDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    extractor = FeatureExtractor().to(args.device).eval()

    count = 0
    feature_sum = None
    feature_sq_sum = None
    with torch.inference_mode():
        for images in loader:
            features = extractor(images.to(args.device, non_blocking=True)).double()
            batch_sum = features.sum(dim=0)
            batch_sq_sum = features.square().sum(dim=0)
            feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
            feature_sq_sum = (
                batch_sq_sum if feature_sq_sum is None
                else feature_sq_sum + batch_sq_sum
            )
            count += features.shape[0]

    mean = feature_sum / count
    variance = (feature_sq_sum / count - mean.square()).clamp_min(1e-6)
    # A variance floor suppresses unstable responses from nearly constant channels.
    variance = variance.clamp_min(variance.median() * 0.01)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean.float().cpu(),
            "variance": variance.float().cpu(),
            "image_size": 256,
            "backbone": "resnet18",
            "weights": "IMAGENET1K_V1",
            "feature_layers": ["layer2", "layer3"],
            "training_images": count,
        },
        args.model_out,
    )
    print(f"Trained on {count} images; saved model to {args.model_out}")


if __name__ == "__main__":
    main()

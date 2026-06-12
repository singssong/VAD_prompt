#!/usr/bin/env python3
"""Fit a spatial Gaussian model to pretrained CNN features from normal images."""

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


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
FEATURE_DIM = 384
SEED = 7


def image_files(directory: Path) -> list[Path]:
    # Only direct children are valid dataset entries; this excludes notebook metadata.
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, files: list[Path]) -> None:
        self.files = files
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
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB"))


def build_extractor(device: torch.device) -> torch.nn.Module:
    backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
    extractor = create_feature_extractor(
        backbone, return_nodes={"layer2": "layer2", "layer3": "layer3"}
    )
    return extractor.eval().to(device)


def make_projection(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    projection = torch.randn(1536, FEATURE_DIM, generator=generator)
    projection = projection / projection.norm(dim=0, keepdim=True)
    return projection.to(device)


@torch.inference_mode()
def extract_features(
    extractor: torch.nn.Module, images: torch.Tensor, projection: torch.Tensor
) -> torch.Tensor:
    outputs = extractor(images)
    layer2 = F.normalize(outputs["layer2"], dim=1)
    layer3 = F.interpolate(
        F.normalize(outputs["layer3"], dim=1),
        size=layer2.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    features = torch.cat((layer2, layer3), dim=1)
    features = torch.einsum("bchw,cd->bdhw", features, projection)
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    files = image_files(args.train_dir)
    if not files:
        raise RuntimeError(f"No training images found directly under {args.train_dir}")

    device = torch.device(args.device)
    loader = DataLoader(
        ImageDataset(files),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, (len(files) + args.batch_size - 1) // args.batch_size),
        pin_memory=device.type == "cuda",
    )
    extractor = build_extractor(device)
    projection = make_projection(device)

    count = 0
    feature_sum = None
    feature_square_sum = None
    cached_features: list[torch.Tensor] = []
    for images in loader:
        features = extract_features(extractor, images.to(device), projection).cpu()
        cached_features.append(features)
        feature_sum = features.sum(dim=0) if feature_sum is None else feature_sum + features.sum(dim=0)
        feature_square_sum = (
            features.square().sum(dim=0)
            if feature_square_sum is None
            else feature_square_sum + features.square().sum(dim=0)
        )
        count += features.shape[0]

    mean = feature_sum / count
    variance = (feature_square_sum / count - mean.square()).clamp_min(1e-4)

    # Calibrate visualization against normal-data pixel distances.
    normal_values = []
    for features in cached_features:
        distance = ((features - mean).square() / variance).mean(dim=1).sqrt()
        normal_values.append(distance.flatten())
    normal_values_tensor = torch.cat(normal_values)
    map_scale = float(torch.quantile(normal_values_tensor, 0.995).clamp_min(1e-6))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "spatial_diagonal_gaussian",
            "backbone": "wide_resnet50_2",
            "feature_nodes": ["layer2", "layer3"],
            "input_size": 256,
            "feature_dim": FEATURE_DIM,
            "projection": projection.cpu(),
            "mean": mean,
            "variance": variance,
            "map_scale": map_scale,
            "num_train_images": count,
        },
        args.model_out,
    )
    print(f"Saved model to {args.model_out} using {count} normal images")


if __name__ == "__main__":
    main()

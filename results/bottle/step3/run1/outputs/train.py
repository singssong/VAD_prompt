#!/usr/bin/env python3
"""Fit spatial normal-feature statistics from normal training images."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_files(root: Path) -> list[Path]:
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    ]
    if not files:
        raise RuntimeError(f"No images found in {root}")
    return sorted(files)


class ImageDataset(Dataset):
    def __init__(self, files: list[Path]):
        self.files = files
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.files[index]) as image:
            image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std


def make_extractor(device: torch.device) -> torch.nn.Module:
    backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
    extractor = create_feature_extractor(
        backbone, return_nodes={"layer1": "layer1", "layer2": "layer2", "layer3": "layer3"}
    )
    return extractor.eval().to(device)


def concatenate_features(features: dict[str, torch.Tensor]) -> torch.Tensor:
    target_size = features["layer2"].shape[-2:]
    maps = []
    for name in ("layer1", "layer2", "layer3"):
        feature = features[name]
        if feature.shape[-2:] != target_size:
            feature = F.interpolate(feature, target_size, mode="bilinear", align_corners=False)
        maps.append(feature)
    return torch.cat(maps, dim=1)


@torch.inference_mode()
def extract_projected(
    extractor: torch.nn.Module,
    loader: DataLoader,
    projection: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batches = []
    for images in loader:
        raw = concatenate_features(extractor(images.to(device, non_blocking=True)))
        raw = F.avg_pool2d(raw, kernel_size=3, stride=1, padding=1)
        projected = torch.einsum("oc,bchw->bohw", projection, raw)
        batches.append(projected.cpu())
    return torch.cat(batches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    files = image_files(args.train_dir)
    loader = DataLoader(
        ImageDataset(files),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, len(files)),
        pin_memory=device.type == "cuda",
    )

    extractor = make_extractor(device)
    with torch.inference_mode():
        sample = next(iter(loader))[:1].to(device)
        raw_channels = concatenate_features(extractor(sample)).shape[1]

    generator = torch.Generator(device=device).manual_seed(args.seed)
    projection = torch.randn(
        args.projection_dim, raw_channels, generator=generator, device=device
    ) / np.sqrt(args.projection_dim)
    features = extract_projected(extractor, loader, projection, device)
    mean = features.mean(dim=0)
    variance = features.var(dim=0, unbiased=False).clamp_min(1e-4)

    train_maps = ((features - mean).square() / variance).mean(dim=1).sqrt()
    calibration_low = float(torch.quantile(train_maps, 0.50))
    calibration_high = float(torch.quantile(train_maps, 0.995))
    if calibration_high <= calibration_low:
        calibration_high = calibration_low + 1.0

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "diagonal_spatial_gaussian",
            "backbone": "wide_resnet50_2_imagenet1k_v2",
            "image_size": 256,
            "feature_grid": list(mean.shape[-2:]),
            "projection": projection.cpu(),
            "mean": mean,
            "variance": variance,
            "calibration_low": calibration_low,
            "calibration_high": calibration_high,
        },
        args.model_out,
    )
    metadata = {
        "training_images": len(files),
        "model": str(args.model_out),
        "device": str(device),
        "feature_shape": list(mean.shape),
    }
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

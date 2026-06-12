#!/usr/bin/env python3
"""Train-only PatchCore-style image anomaly scoring."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path, transform) -> None:
        self.paths = sorted(
            path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


class PatchFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = self.layer1(features)
        layer2 = self.layer2(features)
        layer3 = self.layer3(layer2)

        # Local averaging makes patch descriptors less sensitive to individual fibers.
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        layer2 = F.normalize(layer2, dim=1)
        layer3 = F.normalize(layer3, dim=1)
        patches = torch.cat((layer2, layer3), dim=1)
        return patches.permute(0, 2, 3, 1).flatten(1, 2)


def make_projection(input_dim: int, output_dim: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(1234)
    projection = torch.randn(input_dim, output_dim, generator=generator, device=device)
    return projection / np.sqrt(output_dim)


@torch.inference_mode()
def build_memory_bank(
    loader: DataLoader,
    extractor: nn.Module,
    projection: torch.Tensor,
    patches_per_image: int,
    device: torch.device,
) -> torch.Tensor:
    banks = []
    generator = torch.Generator(device=device).manual_seed(5678)
    for images, _ in loader:
        patches = extractor(images.to(device, non_blocking=True))
        patches = F.normalize(patches @ projection, dim=-1)
        for image_patches in patches:
            count = min(patches_per_image, image_patches.shape[0])
            indices = torch.randperm(image_patches.shape[0], generator=generator, device=device)[:count]
            banks.append(image_patches[indices].cpu())
    return torch.cat(banks).contiguous()


@torch.inference_mode()
def score_images(
    loader: DataLoader,
    extractor: nn.Module,
    projection: torch.Tensor,
    memory_bank: torch.Tensor,
    top_fraction: float,
    device: torch.device,
) -> list[tuple[str, float]]:
    results = []
    bank = memory_bank.to(device=device, dtype=torch.float16)
    bank_t = bank.T.contiguous()
    for images, names in loader:
        patches = extractor(images.to(device, non_blocking=True))
        patches = F.normalize(patches @ projection, dim=-1).to(torch.float16)
        # Unit-normalized squared Euclidean distance is 2 - 2*cosine_similarity.
        max_similarity = patches @ bank_t
        patch_scores = (2.0 - 2.0 * max_similarity.amax(dim=-1)).clamp_min_(0).float().sqrt_()
        top_k = max(1, round(patch_scores.shape[1] * top_fraction))
        image_scores = patch_scores.topk(top_k, dim=1).values.mean(dim=1)
        results.extend((name, float(score)) for name, score in zip(names, image_scores.cpu()))
    return results


def write_scores(path: Path, scores: list[tuple[str, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "anomaly_score"))
        writer.writerows((name, f"{score:.8f}") for name, score in scores)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patches-per-image", type=int, default=50)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("This configuration requires CUDA")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    transform = weights.transforms(crop_size=256, resize_size=256)
    train_dataset = ImageDataset(args.train_dir, transform)
    test_dataset = ImageDataset(args.test_dir, transform)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    extractor = PatchFeatureExtractor().eval().to(device)
    projection = make_projection(1536, args.projection_dim, device)
    memory_bank = build_memory_bank(
        train_loader, extractor, projection, args.patches_per_image, device
    )
    scores = score_images(
        test_loader, extractor, projection, memory_bank, args.top_fraction, device
    )
    write_scores(args.output, scores)
    print(f"Wrote {len(scores)} scores to {args.output}")
    print(f"Memory bank: {len(memory_bank)} patches")
    print("Method: PatchCore-style patch nearest-neighbor anomaly scoring")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

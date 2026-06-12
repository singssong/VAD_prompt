#!/usr/bin/env python3
"""Score test images with an unsupervised PatchCore-style memory bank."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    ]


class ImageDataset(Dataset):
    def __init__(self, files: list[Path], transform) -> None:
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.files[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


def make_projection(input_dim: int, output_dim: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(0)
    projection = torch.randn(
        input_dim, output_dim, generator=generator, device=device
    )
    return projection / output_dim**0.5


@torch.inference_mode()
def patch_features(
    extractor: torch.nn.Module,
    images: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    features = extractor(images)
    layer2 = features["layer2"]
    layer3 = F.interpolate(
        features["layer3"],
        size=layer2.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    combined = torch.cat((layer2, layer3), dim=1)
    combined = F.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)
    patches = combined.permute(0, 2, 3, 1).flatten(0, 2)
    patches = patches @ projection
    return F.normalize(patches, dim=1)


@torch.inference_mode()
def build_memory_bank(
    loader: DataLoader,
    extractor: torch.nn.Module,
    projection: torch.Tensor,
    device: torch.device,
    bank_size: int,
) -> torch.Tensor:
    batches = []
    for images, _ in loader:
        batches.append(patch_features(extractor, images.to(device), projection).cpu())
    all_patches = torch.cat(batches)
    generator = torch.Generator().manual_seed(0)
    if len(all_patches) > bank_size:
        indices = torch.randperm(len(all_patches), generator=generator)[:bank_size]
        all_patches = all_patches[indices]
    return all_patches.to(device)


@torch.inference_mode()
def score_images(
    loader: DataLoader,
    extractor: torch.nn.Module,
    projection: torch.Tensor,
    memory_bank: torch.Tensor,
    device: torch.device,
    top_k: int,
) -> list[tuple[str, float]]:
    scored = []
    for images, names in loader:
        batch_size = images.shape[0]
        patches = patch_features(extractor, images.to(device), projection)
        patches_per_image = patches.shape[0] // batch_size

        # Unit-normalized vectors make squared Euclidean distance equal 2 - 2*cosine.
        nearest_distances = []
        for chunk in patches.split(1024):
            similarities = chunk @ memory_bank.T
            nearest_distances.append(2.0 - 2.0 * similarities.max(dim=1).values)
        nearest = torch.cat(nearest_distances).clamp_min_(0).sqrt_()
        nearest = nearest.reshape(batch_size, patches_per_image)
        image_scores = nearest.topk(min(top_k, patches_per_image), dim=1).values.mean(1)
        scored.extend(zip(names, image_scores.cpu().tolist()))
    return scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bank-size", type=int, default=30000)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files = image_files(args.train_dir)
    test_files = image_files(args.test_dir)
    if not train_files or not test_files:
        raise RuntimeError("Both train and test directories must contain images")

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    transform = weights.transforms()
    train_loader = DataLoader(
        ImageDataset(train_files, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ImageDataset(test_files, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    backbone = wide_resnet50_2(weights=weights).to(device).eval()
    extractor = create_feature_extractor(
        backbone, return_nodes={"layer2": "layer2", "layer3": "layer3"}
    )
    projection = make_projection(1536, args.projection_dim, device)
    memory_bank = build_memory_bank(
        train_loader, extractor, projection, device, args.bank_size
    )
    scores = score_images(
        test_loader,
        extractor,
        projection,
        memory_bank,
        device,
        args.top_k,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "anomaly_score"])
        writer.writerows((name, f"{score:.8f}") for name, score in scores)

    print(f"Scored {len(scores)} images -> {args.output}")
    print("Method: PatchCore-style nearest-neighbor patch memory bank")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2 (layer2 + layer3)")


if __name__ == "__main__":
    main()

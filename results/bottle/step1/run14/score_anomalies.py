#!/usr/bin/env python3
"""Score test images with a PatchCore-style pretrained feature memory bank."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, path.name


class FeatureExtractor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool, model.layer1
        )
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        layer1 = self.stem(images)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)

        # Local averaging gives each descriptor a small amount of spatial context.
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )

        # Normalize each scale separately so neither backbone stage dominates.
        layer2 = F.normalize(layer2, dim=1)
        layer3 = F.normalize(layer3, dim=1)
        return torch.cat((layer2, layer3), dim=1)


def foreground_mask(images: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    # The acquisition background is near-white. Keep bottle and foreign-object patches.
    darkness = (images.mean(dim=1, keepdim=True) < 0.94).float()
    mask = F.adaptive_max_pool2d(darkness, grid_size)
    return mask[:, 0] > 0


@torch.inference_mode()
def build_memory_bank(
    loader: DataLoader,
    extractor: FeatureExtractor,
    device: torch.device,
    max_patches: int,
    seed: int,
) -> torch.Tensor:
    descriptors = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        features = extractor(images)
        mask = foreground_mask(images, features.shape[-2:])
        patches = features.permute(0, 2, 3, 1)[mask]
        descriptors.append(patches.cpu())

    memory = torch.cat(descriptors)
    if len(memory) > max_patches:
        generator = torch.Generator().manual_seed(seed)
        selected = torch.randperm(len(memory), generator=generator)[:max_patches]
        memory = memory[selected]
    return memory.contiguous()


def nearest_distances(
    queries: torch.Tensor, memory: torch.Tensor, memory_chunk: int
) -> torch.Tensor:
    minima = torch.full(
        (len(queries),), float("inf"), dtype=queries.dtype, device=queries.device
    )
    for start in range(0, len(memory), memory_chunk):
        distances = torch.cdist(queries, memory[start : start + memory_chunk])
        minima = torch.minimum(minima, distances.min(dim=1).values)
    return minima


@torch.inference_mode()
def score_images(
    loader: DataLoader,
    extractor: FeatureExtractor,
    memory_cpu: torch.Tensor,
    device: torch.device,
    memory_chunk: int,
) -> list[tuple[str, float]]:
    memory = memory_cpu.to(device)
    results = []
    for images, names in loader:
        images = images.to(device, non_blocking=True)
        features = extractor(images)
        masks = foreground_mask(images, features.shape[-2:])
        patch_grid = features.permute(0, 2, 3, 1)

        for index, name in enumerate(names):
            queries = patch_grid[index][masks[index]]
            distances = nearest_distances(queries, memory, memory_chunk)
            top_count = max(1, round(0.01 * len(distances)))
            score = distances.topk(top_count).values.mean().item()
            results.append((name, score))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-patches", type=int, default=30_000)
    parser.add_argument("--memory-chunk", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    transform = weights.transforms(crop_size=256, resize_size=256)

    train_paths = image_files(args.train_dir)
    test_paths = image_files(args.test_dir)
    if not train_paths or not test_paths:
        raise RuntimeError("Both train and test directories must contain images")

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        ImageDataset(train_paths, transform), shuffle=False, **loader_options
    )
    test_loader = DataLoader(
        ImageDataset(test_paths, transform), shuffle=False, **loader_options
    )

    extractor = FeatureExtractor().to(device)
    memory = build_memory_bank(
        train_loader, extractor, device, args.max_patches, args.seed
    )
    scores = score_images(
        test_loader, extractor, memory, device, args.memory_chunk
    )

    with args.output.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("filename", "anomaly_score"))
        for filename, score in scores:
            writer.writerow((filename, f"{score:.8f}"))

    metadata = {
        "method": "PatchCore-style nearest-neighbor patch memory bank",
        "backbone": "Wide ResNet-50-2 (ImageNet-1K V2)",
        "feature_layers": ["layer2", "layer3"],
        "train_images": len(train_paths),
        "test_images": len(test_paths),
        "memory_patches": len(memory),
        "image_score": "mean of top 1% foreground patch distances",
        "device": str(device),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {len(scores)} scores to {args.output}")


if __name__ == "__main__":
    main()

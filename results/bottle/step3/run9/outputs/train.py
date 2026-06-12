#!/usr/bin/env python3
"""Train a feature-memory anomaly detector using normal images only."""

import argparse
import math
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
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def image_paths(directory: Path) -> list[Path]:
    paths = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise RuntimeError(f"No images found in {directory}")
    return paths


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD, path.name


def build_extractor(device: torch.device) -> torch.nn.Module:
    backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
    extractor = create_feature_extractor(
        backbone, return_nodes={"layer2": "layer2", "layer3": "layer3"}
    )
    return extractor.eval().to(device)


def make_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    projection /= math.sqrt(output_dim)
    return projection


@torch.inference_mode()
def extract_embeddings(
    extractor: torch.nn.Module,
    images: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    features = extractor(images)
    layer2 = F.avg_pool2d(features["layer2"], kernel_size=3, stride=1, padding=1)
    layer3 = F.avg_pool2d(features["layer3"], kernel_size=3, stride=1, padding=1)
    layer3 = F.interpolate(
        layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
    )
    patches = torch.cat([layer2, layer3], dim=1).permute(0, 2, 3, 1)
    return patches @ projection


@torch.inference_mode()
def nearest_distances(
    queries: torch.Tensor, memory: torch.Tensor, chunk_size: int = 4096
) -> torch.Tensor:
    query_sq = (queries * queries).sum(dim=1, keepdim=True)
    minimum = torch.full(
        (queries.shape[0],), float("inf"), device=queries.device
    )
    for start in range(0, memory.shape[0], chunk_size):
        chunk = memory[start : start + chunk_size]
        distances_sq = (
            query_sq
            + (chunk * chunk).sum(dim=1).unsqueeze(0)
            - 2.0 * queries @ chunk.T
        )
        minimum = torch.minimum(minimum, distances_sq.min(dim=1).values)
    return minimum.clamp_min_(0).sqrt_()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-size", type=int, default=30000)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    paths = image_paths(args.train_dir)
    shuffled = paths.copy()
    random.shuffle(shuffled)
    calibration_count = max(1, round(len(shuffled) * 0.1))
    calibration_paths = shuffled[:calibration_count]
    memory_paths = shuffled[calibration_count:]

    extractor = build_extractor(device)
    projection = make_projection(1536, args.projection_dim, args.seed).to(device)
    loader = DataLoader(
        ImageDataset(memory_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    all_embeddings = []
    for images, _ in loader:
        embeddings = extract_embeddings(
            extractor, images.to(device, non_blocking=True), projection
        )
        all_embeddings.append(embeddings.reshape(-1, args.projection_dim).cpu())
    candidates = torch.cat(all_embeddings)
    generator = torch.Generator().manual_seed(args.seed)
    selected = torch.randperm(candidates.shape[0], generator=generator)[
        : min(args.memory_size, candidates.shape[0])
    ]
    memory = candidates[selected].contiguous().to(device)

    calibration_loader = DataLoader(
        ImageDataset(calibration_paths),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    calibration_distances = []
    for images, _ in calibration_loader:
        patches = extract_embeddings(
            extractor, images.to(device, non_blocking=True), projection
        ).reshape(-1, args.projection_dim)
        calibration_distances.append(nearest_distances(patches, memory).cpu())
    calibration_distances = torch.cat(calibration_distances)
    map_scale = max(float(torch.quantile(calibration_distances, 0.995)), 1e-6)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory": memory.cpu(),
            "projection": projection.cpu(),
            "map_scale": map_scale,
            "input_size": 256,
            "feature_grid": 32,
            "backbone": "wide_resnet50_2",
            "weights": "IMAGENET1K_V2",
            "seed": args.seed,
        },
        args.model_path,
    )
    print(
        f"Saved {memory.shape[0]} normal patch embeddings to {args.model_path}; "
        f"map scale={map_scale:.6f}"
    )


if __name__ == "__main__":
    main()

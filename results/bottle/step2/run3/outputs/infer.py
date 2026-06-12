#!/usr/bin/env python3
"""Score test images with a trained PatchCore-style detector."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights

from train import FeatureExtractor, IMAGE_EXTENSIONS


def image_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class NamedImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.transform = Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms(
            crop_size=256, resize_size=256
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), path.name


def nearest_distances(
    queries: torch.Tensor, memory: torch.Tensor, memory_chunk: int = 2500
) -> torch.Tensor:
    """Cosine distance to the nearest memory vector without a huge distance matrix."""
    best = torch.full((len(queries),), float("inf"), device=queries.device)
    for start in range(0, len(memory), memory_chunk):
        similarities = queries @ memory[start:start + memory_chunk].T
        best = torch.minimum(best, 1.0 - similarities.max(dim=1).values)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    paths = image_files(args.test_dir)
    if not paths:
        raise RuntimeError(f"No test images found in {args.test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)
    memory = checkpoint["memory_bank"].to(device=device, dtype=torch.float32)
    memory = F.normalize(memory, dim=1)
    channel_indices = checkpoint["channel_indices"].to(device)
    extractor = FeatureExtractor().eval().to(device)
    loader = DataLoader(
        NamedImageDataset(paths), batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == "cuda"
    )

    raw_maps: dict[str, np.ndarray] = {}
    image_scores: dict[str, float] = {}
    with torch.inference_mode():
        for images, names in loader:
            features = extractor(images.to(device, non_blocking=True))
            features = F.normalize(features[:, channel_indices], dim=1)
            batch, channels, height, width = features.shape
            patches = features.permute(0, 2, 3, 1).reshape(-1, channels)
            distances = nearest_distances(patches, memory).reshape(batch, 1, height, width)
            maps = F.interpolate(distances, size=(256, 256), mode="bilinear", align_corners=False)
            maps = F.avg_pool2d(maps, kernel_size=9, stride=1, padding=4)
            maps_np = maps[:, 0].cpu().numpy()
            for name, anomaly_map in zip(names, maps_np):
                raw_maps[name] = anomaly_map
                tail_count = max(1, anomaly_map.size // 100)
                top_tail = np.partition(anomaly_map.ravel(), -tail_count)[-tail_count:]
                image_scores[name] = float(top_tail.mean())

    # One shared unsupervised scale preserves score comparability across PNGs.
    all_values = np.concatenate([raw_maps[name].ravel() for name in sorted(raw_maps)])
    low, high = np.percentile(all_values, [1.0, 99.5])
    scale = max(float(high - low), 1e-8)
    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    for name, anomaly_map in raw_maps.items():
        normalized = np.clip((anomaly_map - low) / scale, 0.0, 1.0)
        Image.fromarray(np.round(normalized * 255).astype(np.uint8), mode="L").save(pixel_dir / name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(image_scores.items())), handle, indent=2)
        handle.write("\n")
    print(f"Scored {len(image_scores)} images using {device}.")


if __name__ == "__main__":
    main()

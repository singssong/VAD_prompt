#!/usr/bin/env python3
"""Score test images against the trained normal patch memory bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from train import ImageDataset, PatchFeatureExtractor


def nearest_memory_distance(
    queries: torch.Tensor,
    memory: torch.Tensor,
    memory_chunk_size: int,
) -> torch.Tensor:
    """Exact nearest-neighbor L2 distance without materializing all distances."""
    query_norm = (queries * queries).sum(dim=1, keepdim=True)
    nearest_squared = torch.full(
        (queries.shape[0],), float("inf"), device=queries.device
    )
    for start in range(0, memory.shape[0], memory_chunk_size):
        chunk = memory[start:start + memory_chunk_size]
        distances = query_norm + (chunk * chunk).sum(dim=1)[None, :]
        distances.addmm_(queries, chunk.T, beta=1.0, alpha=-2.0)
        nearest_squared = torch.minimum(nearest_squared, distances.min(dim=1).values)
    return nearest_squared.clamp_min_(0).sqrt_()


def save_score_map(score_map: torch.Tensor, output_path: Path) -> None:
    # Fixed log mapping keeps contrast across both subtle and strong anomalies.
    encoded = torch.log1p(score_map) / np.log1p(10000.0)
    encoded = (encoded.clamp(0, 1) * 65535.0).round().to(torch.uint16).cpu().numpy()
    Image.fromarray(encoded).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-chunk-size", type=int, default=7500)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    dataset = ImageDataset(args.test_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    extractor = PatchFeatureExtractor().to(device)
    channel_index = checkpoint["channel_index"].to(device)
    feature_mean = checkpoint["feature_mean"].to(device)
    feature_std = checkpoint["feature_std"].to(device)
    memory = checkpoint["memory_bank"].to(device=device, dtype=torch.float32)

    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    image_scores: dict[str, float] = {}

    with torch.inference_mode():
        for images, names in loader:
            features = extractor(images.to(device, non_blocking=True))
            features = features[:, :, channel_index]
            features = (features - feature_mean) / feature_std
            flat_features = features.reshape(-1, features.shape[-1])
            distances = nearest_memory_distance(
                flat_features, memory, args.memory_chunk_size
            )
            patch_maps = distances.reshape(images.shape[0], 1, 16, 16)
            pixel_maps = F.interpolate(
                patch_maps, size=(256, 256), mode="bilinear", align_corners=False
            )[:, 0]
            pixel_maps = F.avg_pool2d(
                pixel_maps[:, None], kernel_size=9, stride=1, padding=4
            )[:, 0]

            for name, score_map in zip(names, pixel_maps):
                flat = score_map.flatten()
                tail_size = max(1, int(np.ceil(flat.numel() * 0.01)))
                image_scores[name] = float(torch.topk(flat, tail_size).values.mean())
                save_score_map(score_map, pixel_dir / name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(image_scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(image_scores)} images using {device}.")


if __name__ == "__main__":
    main()

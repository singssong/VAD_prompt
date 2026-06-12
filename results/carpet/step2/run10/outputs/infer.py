#!/usr/bin/env python3
"""Score images with a trained PatchCore-style patch memory bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train import FeatureExtractor, image_paths, load_image


def nearest_distances(
    queries: torch.Tensor, bank: torch.Tensor, chunk_size: int = 4096
) -> torch.Tensor:
    """Cosine distance to the nearest memory patch, chunked to bound memory."""
    best = torch.full((queries.shape[0],), float("inf"), device=queries.device)
    for start in range(0, bank.shape[0], chunk_size):
        similarities = queries @ bank[start:start + chunk_size].T
        best = torch.minimum(best, 1.0 - similarities.max(dim=1).values)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    bank = checkpoint["memory_bank"].to(device=device, dtype=torch.float32)
    mean = checkpoint["feature_mean"].to(device)
    std = checkpoint["feature_std"].to(device)
    extractor = FeatureExtractor().to(device)
    paths = image_paths(args.test_dir)
    if not paths:
        raise RuntimeError(f"No test images found in {args.test_dir}")

    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}

    with torch.inference_mode():
        for index, path in enumerate(paths, 1):
            features = extractor(load_image(path, device))
            height, width = features.shape[-2:]
            patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
            patches = F.normalize((patches - mean) / std, dim=1)
            distances = nearest_distances(patches, bank)
            patch_map = distances.reshape(1, 1, height, width)
            anomaly_map = F.interpolate(
                patch_map, size=(256, 256), mode="bilinear", align_corners=False
            )
            # Light smoothing suppresses feature-grid interpolation artifacts.
            anomaly_map = F.avg_pool2d(
                F.pad(anomaly_map, (4, 4, 4, 4), mode="reflect"), kernel_size=9, stride=1
            )
            flat = anomaly_map.flatten()
            top_count = max(1, flat.numel() // 100)
            image_score = flat.topk(top_count).values.mean().item()
            scores[path.name] = float(image_score)

            # Keep one shared scale across all images for calibrated pixel scores.
            map_array = anomaly_map[0, 0].cpu().numpy()
            normalized = np.clip(map_array, 0.0, 1.0)
            Image.fromarray(np.round(normalized * 255).astype(np.uint8), mode="L").save(
                pixel_dir / path.name
            )
            if index % 10 == 0 or index == len(paths):
                print(f"Scored images: {index}/{len(paths)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved scores for {len(scores)} images")


if __name__ == "__main__":
    main()

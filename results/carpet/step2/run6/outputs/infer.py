#!/usr/bin/env python3
"""Score test images and write image-level and pixel-level anomaly scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train import FeatureExtractor, descriptors, image_files, nearest_distances


def smooth_and_resize(score_map: torch.Tensor) -> torch.Tensor:
    score_map = score_map[None, None]
    coordinates = torch.arange(-4, 5, dtype=torch.float32)
    kernel = torch.exp(-(coordinates ** 2) / (2 * 2.0 ** 2))
    kernel = kernel / kernel.sum()
    kernel2d = (kernel[:, None] * kernel[None, :])[None, None]
    score_map = F.conv2d(score_map, kernel2d, padding=4)
    return F.interpolate(
        score_map, size=(256, 256), mode="bilinear", align_corners=False
    )[0, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    memory = checkpoint["memory"].float()
    projection = checkpoint["projection"].float().to(device)
    median = checkpoint["calibration_median"]
    scale = checkpoint["calibration_scale"]

    paths = image_files(args.test_dir)
    if not paths:
        raise RuntimeError(f"No test images found in {args.test_dir}")
    extractor = FeatureExtractor().to(device)
    feature_sets = descriptors(extractor, paths, projection, args.batch_size, device)

    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}
    print(f"Scoring {len(paths)} images on {device}...")
    for path, features in zip(paths, feature_sets):
        distances = nearest_distances(features, memory, device)
        side = int(round(len(distances) ** 0.5))
        calibrated = ((distances - median) / scale).clamp_min(0)
        # A top-tail mean is robust to isolated feature noise but responds to defects.
        top_count = max(1, int(0.01 * len(calibrated)))
        image_score = calibrated.topk(top_count).values.mean().item()
        scores[path.name] = float(image_score)

        dense_map = smooth_and_resize(calibrated.reshape(side, side))
        png = (dense_map.clamp(0, 8) * (255.0 / 8.0)).round().byte().numpy()
        Image.fromarray(png, mode="L").save(pixel_dir / path.name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote scores and anomaly maps to {args.output_dir}")


if __name__ == "__main__":
    main()

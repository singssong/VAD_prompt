#!/usr/bin/env python3
"""Score test images with a trained normal feature-memory model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from train import (
    Config,
    ImageDataset,
    MidLevelExtractor,
    list_images,
    score_feature_grid,
)


def normalize(value: float, low: float, high: float) -> float:
    midpoint = 0.5 * (low + high)
    scale = max((high - low) / 4.0, 1e-8)
    logit = float(np.clip((value - midpoint) / scale, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-logit)))


def score_images(
    extractor: MidLevelExtractor,
    loader: DataLoader,
    memory: torch.Tensor,
    channel_mean: torch.Tensor,
    channel_std: torch.Tensor,
    calibration: dict[str, float],
    config: Config,
    output_dir: Path,
    device: torch.device,
) -> dict[str, float]:
    """Extract features, score pixels/images, and write one map per image."""
    pixel_dir = output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}

    with torch.inference_mode():
        for images, filenames in loader:
            grids = extractor(images.to(device))
            for index, filename in enumerate(filenames):
                anomaly_map, raw_score = score_feature_grid(
                    grids[index], memory, channel_mean, channel_std, config
                )
                resized = F.interpolate(
                    anomaly_map[None, None],
                    size=(config.image_size, config.image_size),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                normalized_map = (
                    (resized - calibration["pixel_low"])
                    / max(calibration["pixel_high"] - calibration["pixel_low"], 1e-8)
                ).clamp(0, 1)
                map_image = Image.fromarray(
                    (normalized_map.numpy() * 255.0).round().astype(np.uint8), mode="L"
                )
                map_image.save(pixel_dir / filename)
                scores[filename] = normalize(
                    raw_score, calibration["image_low"], calibration["image_high"]
                )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="data/test_images")
    parser.add_argument("--model-path", default="outputs/model.pt")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)
    config = Config(**checkpoint["config"])
    test_paths = list_images(Path(args.test_dir))
    if not test_paths:
        raise RuntimeError(f"No supported test images found in {args.test_dir}")

    extractor = MidLevelExtractor().to(device).eval()
    loader = DataLoader(
        ImageDataset(test_paths, config.image_size),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    scores = score_images(
        extractor,
        loader,
        checkpoint["memory"].to(device),
        checkpoint["channel_mean"].to(device),
        checkpoint["channel_std"].to(device),
        checkpoint["calibration"],
        config,
        Path(args.output_dir),
        device,
    )
    score_path = Path(args.output_dir) / "image_scores.json"
    score_path.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    print(f"Scored {len(scores)} images; results written to {score_path}")


if __name__ == "__main__":
    main()

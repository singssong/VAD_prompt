#!/usr/bin/env python3
"""Score test images and write image- and pixel-level anomaly results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train import (
    Config,
    FeatureExtractor,
    extract_features,
    image_paths,
    make_loader,
    score_feature_maps,
)


def normalize(values: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Map the training median to 0.1 and its 99.5th percentile to 0.9."""
    center = (low + high) / 2.0
    scale = 2.0 * math.tan(0.4 * math.pi) / max(high - low, 1e-12)
    return 0.5 + torch.atan((values.double() - center) * scale) / math.pi


def score_images(config: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(config.model_path, map_location="cpu", weights_only=False)
    saved = artifact["config"]
    runtime = Config(
        train_dir=config.train_dir,
        test_dir=config.test_dir,
        output_dir=config.output_dir,
        model_path=config.model_path,
        image_size=int(saved["image_size"]),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        selected_channels=int(saved["selected_channels"]),
        covariance_regularization=float(saved["covariance_regularization"]),
        smoothing_sigma=float(saved["smoothing_sigma"]),
        image_top_fraction=float(saved["image_top_fraction"]),
        seed=int(saved["seed"]),
    )

    paths = image_paths(runtime.test_dir)
    extractor = FeatureExtractor().to(device)
    features, names = extract_features(
        extractor,
        make_loader(paths, runtime),
        device,
        artifact["selected_indices"],
    )
    maps, raw_scores = score_feature_maps(
        features,
        artifact["mean"],
        artifact["precision"],
        runtime.smoothing_sigma,
        runtime.image_top_fraction,
        device,
    )

    image_scores = normalize(raw_scores, *artifact["image_score_range"])
    pixel_maps = normalize(maps, *artifact["pixel_score_range"])
    pixel_maps = F.interpolate(
        pixel_maps.unsqueeze(1),
        size=(runtime.image_size, runtime.image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    output_dir = Path(runtime.output_dir)
    pixel_dir = output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, float] = {}
    for name, score, pixel_map in zip(names, image_scores, pixel_maps):
        result[name] = float(score)
        output = np.rint(pixel_map.numpy() * 255.0).astype(np.uint8)
        Image.fromarray(output, mode="L").save(pixel_dir / name)

    with (output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(names)} test images using {device}.")
    print(f"Wrote results to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default=Config.test_dir)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--model-path", default=Config.model_path)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score_images(
        Config(
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )

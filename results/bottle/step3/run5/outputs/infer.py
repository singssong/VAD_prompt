#!/usr/bin/env python3
"""Score images with a fitted spatial patch-feature distribution."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from train import FeatureExtractor, ImageDataset, image_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = image_files(args.test_dir)
    if not paths:
        raise RuntimeError(f"No images found in {args.test_dir}")

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    mean = state["mean"].to(args.device)
    variance = state["variance"].to(args.device)
    extractor = FeatureExtractor().to(args.device).eval()
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    maps = []
    with torch.inference_mode():
        for images in loader:
            features = extractor(images.to(args.device, non_blocking=True))
            anomaly = ((features - mean).square() / variance).mean(dim=1, keepdim=True)
            anomaly = F.avg_pool2d(anomaly, kernel_size=3, stride=1, padding=1)
            anomaly = F.interpolate(
                anomaly, size=(256, 256), mode="bilinear", align_corners=False
            )
            maps.extend(anomaly[:, 0].cpu().numpy())

    # Robust top-area aggregation is less sensitive to a single noisy patch.
    scores = {}
    for path, anomaly_map in zip(paths, maps):
        top_count = max(1, anomaly_map.size // 100)
        image_score = float(np.partition(anomaly_map.ravel(), -top_count)[-top_count:].mean())
        scores[path.name] = image_score

    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    all_values = np.concatenate([anomaly_map.ravel() for anomaly_map in maps])
    low, high = np.percentile(all_values, [1.0, 99.5])
    scale = max(float(high - low), 1e-12)
    for path, anomaly_map in zip(paths, maps):
        normalized = np.clip((anomaly_map - low) / scale, 0.0, 1.0)
        output = Image.fromarray(np.round(normalized * 65535).astype(np.uint16))
        output.save(pixel_dir / path.name)

    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Scored {len(paths)} images; outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()

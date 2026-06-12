#!/usr/bin/env python3
"""Score test images with the trained spatial feature anomaly detector."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from train import ImageDataset, ResNet18Features, anomaly_maps, gaussian_kernel, image_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = image_files(args.test_dir)
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise RuntimeError("Test image basenames are not unique")

    device = torch.device(args.device)
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)
    mean = checkpoint["mean"].to(device)
    denominator = checkpoint["denominator"].to(device)
    map_low = float(checkpoint["map_low"])
    map_high = float(checkpoint["map_high"])

    extractor = ResNet18Features().to(device).eval()
    kernel = gaussian_kernel().to(device)
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}

    with torch.inference_mode():
        for images, batch_names in loader:
            features = extractor(images.to(device, non_blocking=True))
            maps = anomaly_maps(features, mean, denominator, kernel)
            maps = F.interpolate(maps, size=(256, 256), mode="bilinear", align_corners=False)

            # A top-1% mean is robust to isolated noise while retaining small defects.
            flattened = maps.flatten(start_dim=1)
            top_count = max(1, int(flattened.shape[1] * 0.01))
            image_scores = flattened.topk(top_count, dim=1).values.mean(dim=1)

            encoded = ((maps - map_low) / (map_high - map_low)).clamp(0, 1)
            encoded = (encoded.mul(255).round().byte().cpu().numpy())
            for index, name in enumerate(batch_names):
                scores[name] = float(image_scores[index].item())
                Image.fromarray(encoded[index, 0], mode="L").save(pixel_dir / name)

    ordered_scores = {name: scores[name] for name in sorted(scores)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "image_scores.json"
    score_path.write_text(json.dumps(ordered_scores, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(ordered_scores)} image scores and pixel maps to {args.output_dir}")


if __name__ == "__main__":
    main()

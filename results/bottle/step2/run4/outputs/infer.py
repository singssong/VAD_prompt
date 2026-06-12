#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from model import FeatureExtractor, ImageDataset, anomaly_map, image_score


def parse_args():
    parser = argparse.ArgumentParser(description="Score images with the fitted model.")
    parser.add_argument("--test-dir", default="./data/test_images")
    parser.add_argument("--model", default="./outputs/model.pt")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    mean = state["mean"][None].to(device)
    variance = state["variance"][None].to(device)
    mask = state["foreground_mask"]
    map_scale = float(state["map_scale"])

    dataset = ImageDataset(args.test_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().to(device)
    output_dir = Path(args.output_dir)
    pixel_dir = output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}

    with torch.inference_mode():
        for images, _, names in loader:
            features = extractor(images.to(device, non_blocking=True))
            maps = anomaly_map(features, mean, variance).cpu() * mask
            for score_map, name in zip(maps, names):
                scores[name] = float(image_score(score_map, mask).item())
                # A single global normal-data calibration preserves cross-image meaning.
                png = (score_map / map_scale * 128.0).clamp(0, 255).byte().numpy()
                Image.fromarray(png, mode="L").save(pixel_dir / name)

    with (output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images on {device}; results saved to {output_dir}")


if __name__ == "__main__":
    main()

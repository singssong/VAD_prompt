#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from common import FeatureExtractor, ImageDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Score images with a patch prototype model.")
    parser.add_argument("--test-dir", default="./data/test_images")
    parser.add_argument("--model", default="./outputs/model.pt")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)

    extractor = FeatureExtractor()
    extractor.load_state_dict(checkpoint["extractor"])
    extractor.eval().requires_grad_(False).to(device)
    prototypes = checkpoint["prototypes"].to(device)
    low = checkpoint["calibration_low"]
    high = checkpoint["calibration_high"]

    dataset = ImageDataset(args.test_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    output_dir = Path(args.output_dir)
    pixel_dir = output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}

    with torch.inference_mode():
        for images, names in loader:
            feature_map = extractor(images.to(device))
            batch, channels, height, width = feature_map.shape
            patches = feature_map.permute(0, 2, 3, 1).reshape(-1, channels)
            distances = 1.0 - (patches @ prototypes.T).max(dim=1).values
            maps = distances.reshape(batch, 1, height, width)

            # Smooth on the patch grid, then bilinearly produce a dense 256x256 map.
            maps = F.avg_pool2d(maps, kernel_size=3, stride=1, padding=1)
            maps = F.interpolate(
                maps, size=(256, 256), mode="bilinear", align_corners=False
            )
            flat_maps = maps.flatten(1)
            top_count = max(1, int(flat_maps.shape[1] * 0.01))
            image_scores = flat_maps.topk(top_count, dim=1).values.mean(dim=1)

            visual = ((maps - low) / max(high - low, 1e-12)).clamp(0, 1)
            visual = (visual.mul(255).round().byte().cpu().numpy()[:, 0])
            for name, score, pixel_map in zip(names, image_scores, visual):
                scores[name] = float(score.item())
                Image.fromarray(pixel_map, mode="L").save(pixel_dir / name)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"scored {len(scores)} images")


if __name__ == "__main__":
    main()


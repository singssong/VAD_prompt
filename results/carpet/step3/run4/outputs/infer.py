#!/usr/bin/env python3
"""Score test images against a trained normal-patch memory bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from train import FeatureExtractor, ImageDataset, image_paths, nearest_distances, project_patches


def gaussian_kernel(size=9, sigma=2.0, device=None):
    coordinates = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    kernel_1d = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    return (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, size, size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="./data/test_images")
    parser.add_argument("--model", default="./outputs/model.pt")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    paths = image_paths(args.test_dir)
    if not paths:
        raise RuntimeError(f"No test images found in {args.test_dir}")
    output_dir = Path(args.output_dir)
    pixel_dir = output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    extractor = FeatureExtractor().eval().to(device)
    projection = state["projection"].to(device)
    bank = state["memory_bank"].to(device)
    kernel = gaussian_kernel(device=device)
    loader = DataLoader(
        ImageDataset(paths), batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda"
    )

    scores = {}
    low, high = state["map_low"], state["map_high"]
    with torch.inference_mode():
        for images, names in loader:
            batch_size = len(images)
            features = extractor(images.to(device, non_blocking=True))
            queries = project_patches(features, projection)
            distances = nearest_distances(queries, bank)
            maps = distances.view(batch_size, 1, state["grid_size"], state["grid_size"])
            maps = F.interpolate(maps, size=(256, 256), mode="bilinear", align_corners=False)
            maps = F.conv2d(maps, kernel, padding=kernel.shape[-1] // 2)

            flattened = maps.flatten(1)
            top_count = max(1, round(flattened.shape[1] * 0.01))
            image_scores = flattened.topk(top_count, dim=1).values.mean(dim=1)
            calibrated = ((maps - low) / (high - low)).clamp(0, 1)

            for index, name in enumerate(names):
                scores[name] = float(image_scores[index].cpu())
                pixels = calibrated[index, 0].mul(65535).round().to(torch.uint16).cpu().numpy()
                Image.fromarray(pixels).save(pixel_dir / name)

    with (output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images into {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score images using a trained normal patch-feature memory bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from train import FeatureExtractor, ImageDataset


def nearest_distances(queries, memory, query_chunk=1024, memory_chunk=4096):
    results = []
    for query in queries.split(query_chunk):
        best = torch.full((len(query),), float("inf"), device=query.device)
        for reference in memory.split(memory_chunk):
            best = torch.minimum(best, torch.cdist(query, reference).amin(dim=1))
        results.append(best)
    return torch.cat(results)


def gaussian_kernel(device, dtype, size=9, sigma=2.0):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :])[None, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="./data/test_images")
    parser.add_argument("--model", default="./outputs/model.pt")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model = FeatureExtractor(checkpoint["channels_per_level"]).eval().to(device)
    memory = checkpoint["memory_bank"].to(device)
    dataset = ImageDataset(args.test_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(4, len(dataset)), pin_memory=device.type == "cuda",
    )
    kernel = gaussian_kernel(device, torch.float32)
    maps = {}
    scores = {}

    with torch.inference_mode():
        for images, names in loader:
            features = model(images.to(device))
            batch, channels, height, width = features.shape
            flat = features.permute(0, 2, 3, 1).reshape(-1, channels)
            distances = nearest_distances(flat, memory).reshape(batch, 1, height, width)
            distances = F.interpolate(
                distances, size=(256, 256), mode="bilinear", align_corners=False
            )
            distances = F.conv2d(distances, kernel, padding=kernel.shape[-1] // 2)
            for index, name in enumerate(names):
                anomaly_map = distances[index, 0].cpu().numpy()
                maps[name] = anomaly_map
                top_count = max(1, anomaly_map.size // 100)
                scores[name] = float(
                    np.partition(anomaly_map.ravel(), -top_count)[-top_count:].mean()
                )

    output_dir = Path(args.output_dir)
    pixel_dir = output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    # Encode maps consistently across the test set as 16-bit grayscale PNGs.
    all_values = np.concatenate([value.ravel() for value in maps.values()])
    low = float(all_values.min())
    high = float(np.quantile(all_values, 0.999))
    scale = max(high - low, 1e-12)
    for name, anomaly_map in maps.items():
        encoded = np.clip((anomaly_map - low) / scale, 0.0, 1.0)
        encoded = np.round(encoded * 65535).astype(np.uint16)
        Image.fromarray(encoded).save(pixel_dir / name, format="PNG")

    with (output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images using {device}; outputs saved to {output_dir}.")


if __name__ == "__main__":
    main()

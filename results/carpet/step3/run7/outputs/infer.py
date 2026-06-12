#!/usr/bin/env python3
"""Score test images with the trained normal-patch memory bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from train import FeatureExtractor, ImageDataset, patch_embeddings


def gaussian_kernel(size=9, sigma=2.0, device="cpu"):
    coordinates = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    kernel = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :]).view(1, 1, size, size)


@torch.no_grad()
def nearest_distances(queries, bank, chunk_size=1024):
    flat = queries.reshape(-1, queries.shape[-1])
    distances = []
    for start in range(0, len(flat), chunk_size):
        similarity = flat[start : start + chunk_size] @ bank.T
        distances.append(1.0 - similarity.max(dim=1).values)
    return torch.cat(distances).reshape(queries.shape[:-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    bank = F.normalize(checkpoint["memory_bank"].float(), dim=1).to(device)
    projection = checkpoint["projection"].float().to(device)
    scale = float(checkpoint["calibration_scale"])
    model = FeatureExtractor().to(device)
    dataset = ImageDataset(args.test_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    kernel = gaussian_kernel(device=device)
    scores = {}

    processed = 0
    for images, names in loader:
        embeddings = patch_embeddings(model(images.to(device)), projection)
        maps = nearest_distances(embeddings, bank).unsqueeze(1)
        maps = F.interpolate(maps, size=(256, 256), mode="bilinear", align_corners=False)
        maps = F.conv2d(maps, kernel, padding=kernel.shape[-1] // 2)
        for anomaly_map, name in zip(maps[:, 0], names):
            flat = anomaly_map.flatten()
            tail_count = max(1, int(flat.numel() * 0.01))
            scores[name] = float(torch.topk(flat, tail_count).values.mean().cpu())
            visual = (anomaly_map / scale * 255.0).clamp(0, 255).byte().cpu().numpy()
            Image.fromarray(visual, mode="L").save(pixel_dir / name)
        processed += len(names)
        print(f"\rScoring test images: {processed}/{len(dataset)}", end="")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {len(scores)} image scores and pixel maps to {args.output_dir}")


if __name__ == "__main__":
    main()

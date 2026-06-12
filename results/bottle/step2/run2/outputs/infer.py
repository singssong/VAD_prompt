#!/usr/bin/env python3
"""Score test images with the trained normal feature memory bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from train import FeatureExtractor, ImageDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-chunk", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def nearest_distances(queries, memory, chunk_size):
    distances = []
    for start in range(0, len(queries), chunk_size):
        similarity = queries[start:start + chunk_size] @ memory.T
        distances.append(1.0 - similarity.max(dim=1).values)
    return torch.cat(distances)


def main():
    args = parse_args()
    dataset = ImageDataset(args.test_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=args.device.startswith("cuda")
    )
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)
    memory = checkpoint["memory"].to(args.device, dtype=torch.float32)
    memory = F.normalize(memory, dim=1)
    extractor = FeatureExtractor().to(args.device)

    maps = {}
    image_scores = {}
    with torch.inference_mode():
        for images, names in loader:
            features = extractor(images.to(args.device, non_blocking=True))
            batch, channels, height, width = features.shape
            queries = features.permute(0, 2, 3, 1).reshape(-1, channels)
            patch_scores = nearest_distances(queries, memory, args.query_chunk)
            patch_scores = patch_scores.reshape(batch, 1, height, width)
            dense = F.interpolate(
                patch_scores, size=(256, 256), mode="bilinear", align_corners=False
            ).squeeze(1).cpu().numpy()
            for name, score_map in zip(names, dense):
                maps[name] = score_map
                top_count = max(1, score_map.size // 100)
                image_scores[name] = float(np.partition(score_map.ravel(), -top_count)[-top_count:].mean())

    # One shared robust scale preserves comparability between all output maps.
    all_pixels = np.concatenate([score_map.ravel() for score_map in maps.values()])
    low, high = np.percentile(all_pixels, [1.0, 99.5])
    if high <= low:
        high = low + 1e-6

    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    for name, score_map in maps.items():
        normalized = np.clip((score_map - low) / (high - low), 0.0, 1.0)
        Image.fromarray(np.round(normalized * 255).astype(np.uint8), mode="L").save(pixel_dir / name)

    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(image_scores.items())), handle, indent=2)
        handle.write("\n")
    print(f"Scored {len(image_scores)} images; wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()

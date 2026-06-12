#!/usr/bin/env python3
"""Score images with a trained normal patch memory bank."""

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
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    return parser.parse_args()


def nearest_patch_distances(features, memory, memory_chunk=10000):
    flat = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    best_similarity = torch.full(
        (len(flat),), -float("inf"), dtype=flat.dtype, device=flat.device
    )
    for memory_part in memory.split(memory_chunk):
        best_similarity = torch.maximum(
            best_similarity, (flat @ memory_part.T).max(dim=1).values
        )
    return (1.0 - best_similarity).reshape(features.shape[0], 1, 16, 16)


def main():
    args = parse_args()
    if not 0 < args.top_fraction <= 1:
        raise ValueError("--top-fraction must be in (0, 1]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(args.model, map_location="cpu", weights_only=True)
    memory = artifact["memory_bank"].float().to(device)
    memory = F.normalize(memory, dim=1)

    dataset = ImageDataset(args.test_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().eval().to(device)
    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)

    scores = {}
    path_index = 0
    map_low = float(artifact["map_low"])
    map_high = float(artifact["map_high"])
    with torch.inference_mode():
        for images in loader:
            features = F.normalize(extractor(images.to(device, non_blocking=True)), dim=1)
            maps = nearest_patch_distances(features, memory)
            maps = F.interpolate(maps, size=(256, 256), mode="bilinear", align_corners=False)
            maps = F.avg_pool2d(maps, kernel_size=9, stride=1, padding=4)

            flattened = maps.flatten(1)
            top_count = max(1, round(flattened.shape[1] * args.top_fraction))
            image_scores = flattened.topk(top_count, dim=1).values.mean(dim=1)

            calibrated = ((maps - map_low) / (map_high - map_low)).clamp(0, 1)
            calibrated = (calibrated.mul(65535).round().to(torch.uint16)).cpu().numpy()
            for batch_index, image_score in enumerate(image_scores.cpu().tolist()):
                source_path = dataset.paths[path_index]
                filename = source_path.name
                if filename in scores:
                    raise RuntimeError(f"Duplicate test filename: {filename}")
                scores[filename] = float(image_score)
                Image.fromarray(calibrated[batch_index, 0]).save(pixel_dir / filename)
                path_index += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images on {device}; outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()

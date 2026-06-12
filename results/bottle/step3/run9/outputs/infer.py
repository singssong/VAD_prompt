#!/usr/bin/env python3
"""Score test images with the trained feature-memory anomaly detector."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from train import ImageDataset, build_extractor, extract_embeddings, image_paths, nearest_distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if checkpoint["backbone"] != "wide_resnet50_2":
        raise RuntimeError(f"Unsupported backbone: {checkpoint['backbone']}")

    extractor = build_extractor(device)
    projection = checkpoint["projection"].to(device)
    memory = checkpoint["memory"].to(device)
    map_scale = float(checkpoint["map_scale"])

    paths = image_paths(args.test_dir)
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}

    with torch.inference_mode():
        for images, names in loader:
            embeddings = extract_embeddings(
                extractor, images.to(device, non_blocking=True), projection
            )
            grid_size = embeddings.shape[1:3]
            distances = nearest_distances(
                embeddings.reshape(-1, embeddings.shape[-1]), memory
            )
            patch_map = distances.reshape(1, 1, *grid_size)
            pixel_map = F.interpolate(
                patch_map, size=(256, 256), mode="bilinear", align_corners=False
            )
            pixel_map = F.avg_pool2d(pixel_map, kernel_size=9, stride=1, padding=4)
            flat = pixel_map.flatten()
            top_count = max(1, round(flat.numel() * 0.01))
            image_score = float(torch.topk(flat, top_count).values.mean())

            output_array = (
                (pixel_map[0, 0] / map_scale).clamp(0, 1).mul(255).round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            name = names[0]
            Image.fromarray(output_array, mode="L").save(pixel_dir / name)
            scores[name] = image_score

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images and wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()

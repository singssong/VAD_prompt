#!/usr/bin/env python3
"""Score test images with nearest-normal-patch feature distances."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from train import ImageDataset, PatchFeatureExtractor, projected_patches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bank-chunk-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def nearest_cosine_distance(patches, memory_bank, chunk_size):
    best_similarity = torch.full(
        (len(patches),), -1.0, dtype=patches.dtype, device=patches.device
    )
    for start in range(0, len(memory_bank), chunk_size):
        bank_chunk = memory_bank[start : start + chunk_size]
        similarity = patches @ bank_chunk.T
        best_similarity = torch.maximum(best_similarity, similarity.max(dim=1).values)
    return 1.0 - best_similarity


def gaussian_kernel(size=9, sigma=2.0, device=None):
    axis = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    kernel_1d = torch.exp(-(axis**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    return (kernel_1d[:, None] * kernel_1d[None, :])[None, None]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    memory_bank = checkpoint["memory_bank"].to(device)
    projection = checkpoint["projection"].to(device)

    dataset = ImageDataset(args.test_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    extractor = PatchFeatureExtractor().eval().to(device)
    smoothing_kernel = gaussian_kernel(device=device)

    maps = []
    scores = {}
    file_index = 0
    with torch.inference_mode():
        for batch_index, images in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            features = extractor(images)
            patches = projected_patches(features, projection)
            distances = nearest_cosine_distance(
                patches, memory_bank, args.bank_chunk_size
            )
            batch_maps = distances.reshape(len(images), 1, 32, 32)
            batch_maps = F.interpolate(
                batch_maps, size=(256, 256), mode="bilinear", align_corners=False
            )
            batch_maps = F.conv2d(batch_maps, smoothing_kernel, padding=4)

            for anomaly_map in batch_maps[:, 0]:
                anomaly_map = anomaly_map.clamp_min(0)
                flat = anomaly_map.flatten()
                top_k = max(1, int(0.01 * flat.numel()))
                score = flat.topk(top_k).values.mean().item()
                filename = dataset.files[file_index].name
                scores[filename] = float(score)
                maps.append((filename, anomaly_map.cpu().numpy()))
                file_index += 1
            print(f"\rScoring test images: {batch_index}/{len(loader)}", end="", flush=True)
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    # PNG has no standard floating-point grayscale mode. A shared scale preserves
    # relative anomaly magnitude across every test image in 16-bit grayscale.
    encoding_max = max(float(np.max(anomaly_map)) for _, anomaly_map in maps)
    encoding_max = max(encoding_max, np.finfo(np.float32).eps)
    for filename, anomaly_map in maps:
        encoded = np.round(np.clip(anomaly_map / encoding_max, 0, 1) * 65535).astype(
            np.uint16
        )
        Image.fromarray(encoded).save(pixel_dir / filename)

    metadata = {
        "pixel_png_dtype": "uint16",
        "pixel_png_scale": encoding_max / 65535.0,
        "pixel_png_decode": "raw_png_value * pixel_png_scale",
        "image_score": "mean of highest 1% of raw 256x256 pixel scores",
    }
    with (args.output_dir / "score_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(scores)} image scores and {len(maps)} pixel maps")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score test images with the trained normal-patch memory bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights

from train import FeatureExtractor, IMAGE_SUFFIXES


def nearest_distances(queries, memory, chunk_size):
    results = []
    memory_t = memory.T
    for start in range(0, len(queries), chunk_size):
        query = queries[start : start + chunk_size]
        # Unit vectors make squared Euclidean distance equal to 2 - 2*cosine.
        similarities = query @ memory_t
        results.append(torch.sqrt((2.0 - 2.0 * similarities.max(dim=1).values).clamp_min(0)))
    return torch.cat(results)


def gaussian_kernel(size=21, sigma=4.0, device=None):
    coordinates = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    return (kernel_1d[:, None] * kernel_1d[None, :])[None, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--scores", type=Path, default=Path("outputs/image_scores.json"))
    parser.add_argument("--pixel-dir", type=Path, default=Path("outputs/pixel_scores"))
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()

    files = sorted(
        p
        for p in args.test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise RuntimeError(f"No images found in {args.test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    memory = F.normalize(checkpoint["memory_bank"].to(device), dim=1)
    projection = checkpoint["projection"].to(device)
    extractor = FeatureExtractor().to(device).eval()
    transform = Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms()
    blur = gaussian_kernel(device=device)

    args.pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}
    raw_maps = {}
    with torch.inference_mode():
        for index, path in enumerate(files, 1):
            with Image.open(path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            features = extractor(tensor)
            height, width = features.shape[-2:]
            patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
            queries = F.normalize(patches @ projection, dim=1)
            distances = nearest_distances(queries, memory, args.chunk_size)
            patch_map = distances.reshape(1, 1, height, width)
            pixel_map = F.interpolate(
                patch_map, size=(256, 256), mode="bilinear", align_corners=False
            )
            pixel_map = F.conv2d(pixel_map, blur, padding=blur.shape[-1] // 2)
            flat = pixel_map.flatten()
            top_count = max(1, int(flat.numel() * 0.01))
            image_score = flat.topk(top_count).values.mean().item()
            scores[path.name] = float(image_score)

            raw_maps[path.name] = pixel_map[0, 0].cpu().numpy()
            print(f"Scored {index}/{len(files)}: {path.name}", flush=True)

    # A shared scale preserves pixel-score comparability between test images.
    scale_high = np.percentile(np.stack(list(raw_maps.values())), 99.9)
    for filename, map_array in raw_maps.items():
        normalized = np.clip(map_array / max(scale_high, 1e-8), 0, 1)
        Image.fromarray(np.round(normalized * 255).astype(np.uint8), mode="L").save(
            args.pixel_dir / filename
        )

    args.scores.parent.mkdir(parents=True, exist_ok=True)
    with args.scores.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved {len(scores)} image scores to {args.scores}")


if __name__ == "__main__":
    main()

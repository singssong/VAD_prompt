#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from model import FeatureExtractor, ImageDataset, flatten_features, gaussian_blur


def parse_args():
    parser = argparse.ArgumentParser(description="Score images against a normal feature bank.")
    parser.add_argument("--test-dir", default="./data/test_images")
    parser.add_argument("--model", default="./outputs/model.pt")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--query-chunk-size", type=int, default=512)
    return parser.parse_args()


def nearest_cosine_distance(queries, memory_bank, chunk_size):
    distances = []
    for start in range(0, queries.shape[0], chunk_size):
        query_chunk = queries[start:start + chunk_size]
        best_similarity = torch.matmul(query_chunk, memory_bank.T).amax(dim=1)
        distances.append((1.0 - best_similarity).clamp_min_(0.0))
    return torch.cat(distances)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    memory_bank = checkpoint["memory_bank"].to(device)
    memory_bank = F.normalize(memory_bank, dim=1)

    dataset = ImageDataset(args.test_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().to(device)
    output_dir = Path(args.output_dir)
    map_dir = output_dir / "pixel_scores"
    map_dir.mkdir(parents=True, exist_ok=True)
    scores = {}

    with torch.inference_mode():
        for images, names in loader:
            images = images.to(device, non_blocking=True)
            feature_maps = extractor(images)
            batch_size, _, height, width = feature_maps.shape
            queries = flatten_features(feature_maps).reshape(-1, feature_maps.shape[1])
            patch_scores = nearest_cosine_distance(
                queries, memory_bank, args.query_chunk_size
            )
            patch_scores = patch_scores.view(batch_size, 1, height, width)
            pixel_scores = F.interpolate(
                patch_scores, size=(256, 256), mode="bilinear", align_corners=False
            )
            pixel_scores = gaussian_blur(pixel_scores)

            for name, anomaly_map in zip(names, pixel_scores[:, 0]):
                flat = anomaly_map.flatten()
                top_count = max(1, int(flat.numel() * 0.01))
                image_score = flat.topk(top_count).values.mean().item()
                scores[name] = float(image_score)

                # Distances are cosine distances in [0, 2]; preserve a common scale.
                encoded = (
                    anomaly_map.clamp(0.0, 1.0).mul(65535.0).round()
                    .to(torch.uint16).cpu().numpy()
                )
                Image.fromarray(encoded).save(map_dir / name)
                print(f"{name}: {image_score:.6f}")

    expected_names = {path.name for path in dataset.files}
    if set(scores) != expected_names:
        raise RuntimeError("Not every test image received a score")
    with (output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(scores.items())), handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(scores)} image scores and pixel maps to {output_dir}")


if __name__ == "__main__":
    main()

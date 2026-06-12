#!/usr/bin/env python3
"""Score images with a trained positional PatchCore model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from train import FeatureExtractor, ImageDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("./data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("./outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = ImageDataset(args.test_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, max(0, len(dataset) // 16)),
        pin_memory=args.device.startswith("cuda"),
    )
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    memory = checkpoint["memory"].float().to(args.device)
    memory_norm = (memory * memory).sum(dim=2)
    channel_indices = checkpoint["channel_indices"].to(args.device)
    extractor = FeatureExtractor().to(args.device)

    score_maps = []
    image_scores = []
    with torch.inference_mode():
        for images in loader:
            features = extractor(images.to(args.device, non_blocking=True))
            features = features[:, channel_indices]
            features = features.permute(0, 2, 3, 1).reshape(
                features.shape[0], -1, features.shape[1]
            )
            feature_norm = (features * features).sum(dim=2, keepdim=True)
            distances = feature_norm + memory_norm.unsqueeze(0)
            distances.add_(-2.0 * torch.einsum("bpc,pnc->bpn", features, memory))
            patch_scores = distances.clamp_min_(0).min(dim=2).values.sqrt_()
            grid_h, grid_w = checkpoint["feature_grid"]
            maps = patch_scores.reshape(-1, 1, grid_h, grid_w)
            maps = F.interpolate(
                maps, size=(256, 256), mode="bilinear", align_corners=False
            )[:, 0]
            flat = maps.flatten(1)
            top_count = max(1, int(flat.shape[1] * 0.01))
            scores = flat.topk(top_count, dim=1).values.mean(dim=1)
            score_maps.extend(maps.cpu().numpy())
            image_scores.extend(scores.cpu().tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)

    # One shared scale keeps pixel intensities comparable between test images.
    scale = max(float(np.percentile(np.stack(score_maps), 99.5)), 1e-8)
    for path, score_map in zip(dataset.paths, score_maps):
        encoded = np.clip(score_map / scale * 255.0, 0, 255).astype(np.uint8)
        output = Image.fromarray(encoded, mode="L").filter(
            ImageFilter.GaussianBlur(radius=4)
        )
        output.save(pixel_dir / path.name)

    scores = {
        path.name: float(score)
        for path, score in zip(dataset.paths, image_scores)
    }
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images into {args.output_dir}")


if __name__ == "__main__":
    main()

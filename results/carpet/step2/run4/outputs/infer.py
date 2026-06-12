#!/usr/bin/env python3
"""Score test images with a trained PatchCore-style memory bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train import FeatureExtractor, IMAGE_EXTENSIONS


class TestDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        from torchvision.models import Wide_ResNet50_2_Weights
        self.transform = Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms(
            crop_size=256, resize_size=256
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), path.name


def gaussian_blur(maps, sigma=4.0):
    radius = int(3 * sigma)
    coordinates = torch.arange(-radius, radius + 1, device=maps.device, dtype=maps.dtype)
    kernel = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    maps = F.pad(maps, (radius, radius, 0, 0), mode="reflect")
    maps = F.conv2d(maps, kernel.view(1, 1, 1, -1))
    maps = F.pad(maps, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(maps, kernel.view(1, 1, -1, 1))


def patch_distances(features, bank):
    query_norm = (features * features).sum(dim=1, keepdim=True)
    bank_norm = (bank * bank).sum(dim=1).unsqueeze(0)
    distances = query_norm + bank_norm - 2.0 * features @ bank.T
    return distances.clamp_min_(0).min(dim=1).values.sqrt()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="./data/test_images")
    parser.add_argument("--model", default="./outputs/model.pt")
    parser.add_argument("--scores-out", default="./outputs/image_scores.json")
    parser.add_argument("--pixel-dir", default="./outputs/pixel_scores")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    paths = sorted(
        path for path in Path(args.test_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No test images found in {args.test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    bank = checkpoint["memory_bank"].to(device)
    projection = checkpoint["projection"].to(device)
    grid = checkpoint["feature_grid"]
    position_center = checkpoint["position_center"].to(device)
    position_scale = checkpoint["position_scale"].to(device)
    low, high = checkpoint["map_low"], checkpoint["map_high"]
    model = FeatureExtractor().to(device).eval()
    loader = DataLoader(
        TestDataset(paths), batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda"
    )

    pixel_dir = Path(args.pixel_dir)
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}
    with torch.inference_mode():
        for images, names in loader:
            batch_features = model(images.to(device, non_blocking=True))
            batch_features = batch_features.permute(0, 2, 3, 1)
            for features, name in zip(batch_features, names):
                features = features.reshape(-1, features.shape[-1]) @ projection
                distances = patch_distances(features, bank).reshape(grid, grid)
                distances = ((distances - position_center) / position_scale).clamp_min(0)
                # A small top-area mean is robust to isolated feature noise while
                # retaining sensitivity to compact carpet defects.
                flat_distances = distances.flatten()
                top_count = max(1, round(0.01 * len(flat_distances)))
                image_score = flat_distances.topk(top_count).values.mean().item()
                scores[name] = float(image_score)

                anomaly_map = distances.reshape(1, 1, grid, grid)
                anomaly_map = F.interpolate(
                    anomaly_map, size=(256, 256), mode="bilinear", align_corners=False
                )
                anomaly_map = gaussian_blur(anomaly_map).squeeze()
                normalized = ((anomaly_map - low) / (high - low)).clamp(0, 1)
                pixels = (normalized * 255).round().byte().cpu().numpy()
                Image.fromarray(pixels, mode="L").save(pixel_dir / name)

    scores_path = Path(args.scores_out)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {len(scores)} image scores and pixel maps")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score test images with a trained PaDiM model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur

from train import FeatureExtractor, ImageDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    dataset = ImageDataset(args.test_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    model = FeatureExtractor().to(device)
    indices = checkpoint["channel_indices"].to(device)
    mean = checkpoint["mean"].to(device)
    precision = checkpoint["precision"].to(device)
    low = float(checkpoint["calibration_low"])
    high = float(checkpoint["calibration_high"])

    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    image_scores = {}
    file_index = 0

    with torch.inference_mode():
        for images in loader:
            features = model(images.to(device, non_blocking=True))[:, indices]
            diff = features.flatten(2) - mean.unsqueeze(0)
            scores = torch.einsum("bdl,lde,bel->bl", diff, precision, diff)
            scores = scores.clamp_min_(0).sqrt_().view(-1, 1, 32, 32)
            scores = F.interpolate(scores, size=(256, 256), mode="bilinear", align_corners=False)
            scores = gaussian_blur(scores, kernel_size=[21, 21], sigma=[4.0, 4.0])

            for raw_map in scores[:, 0].cpu():
                path = dataset.files[file_index]
                threshold = torch.quantile(raw_map, 0.995)
                image_score = raw_map[raw_map >= threshold].mean().item()
                image_scores[path.name] = float(image_score)

                normalized = ((raw_map - low) / (high - low)).clamp(0, 1)
                output_map = Image.fromarray(
                    (normalized.numpy() * 255.0).round().astype(np.uint8), mode="L"
                )
                output_map.save(pixel_dir / path.name)
                file_index += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(image_scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(image_scores)} images; outputs: {args.output_dir}")


if __name__ == "__main__":
    main()

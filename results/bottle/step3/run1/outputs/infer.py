#!/usr/bin/env python3
"""Score test images and write image- and pixel-level anomaly scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from train import ImageDataset, concatenate_features, image_files, make_extractor


def gaussian_kernel(size: int = 21, sigma: float = 4.0) -> torch.Tensor:
    coordinates = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d).view(1, 1, size, size)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    projection = checkpoint["projection"].to(device)
    mean = checkpoint["mean"].to(device)
    variance = checkpoint["variance"].to(device)
    low = checkpoint["calibration_low"]
    high = checkpoint["calibration_high"]

    files = image_files(args.test_dir)
    loader = DataLoader(
        ImageDataset(files),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, len(files)),
        pin_memory=device.type == "cuda",
    )
    extractor = make_extractor(device)
    smooth_kernel = gaussian_kernel().to(device)
    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}
    offset = 0

    for images in loader:
        images = images.to(device, non_blocking=True)
        raw = concatenate_features(extractor(images))
        raw = F.avg_pool2d(raw, kernel_size=3, stride=1, padding=1)
        features = torch.einsum("oc,bchw->bohw", projection, raw)
        anomaly = ((features - mean).square() / variance).mean(dim=1, keepdim=True).sqrt()
        anomaly = F.interpolate(anomaly, (256, 256), mode="bilinear", align_corners=False)
        anomaly = F.conv2d(anomaly, smooth_kernel, padding=smooth_kernel.shape[-1] // 2)

        flat = anomaly.flatten(1)
        top_count = max(1, int(flat.shape[1] * 0.01))
        image_scores = flat.topk(top_count, dim=1).values.mean(dim=1)
        encoded = ((anomaly - low) / (high - low)).clamp(0, 1).mul(255).round().byte()

        for index in range(images.shape[0]):
            path = files[offset + index]
            scores[path.name] = float(image_scores[index].cpu())
            Image.fromarray(encoded[index, 0].cpu().numpy(), mode="L").save(
                pixel_dir / path.name
            )
        offset += images.shape[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images into {args.output_dir}")


if __name__ == "__main__":
    main()

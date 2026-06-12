#!/usr/bin/env python3
"""Score test images with the trained spatial nearest-neighbor model."""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from train import (
    CONFIG,
    FeatureExtractor,
    ImageDataset,
    aggregate_maps,
    extract_features,
    image_files,
    score_descriptors,
    smooth_maps,
)


def saturating_normalize(values: torch.Tensor, low: float, scale: float) -> torch.Tensor:
    """Map calibrated nonnegative excess scores monotonically into [0, 1)."""
    return -torch.expm1(-torch.relu(values - low) / scale)


def score_images(
    extractor: FeatureExtractor,
    loader: DataLoader,
    normal_bank: torch.Tensor,
    projection: torch.Tensor,
    calibration: dict[str, float],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Extract test features and produce image and 256x256 pixel scores."""
    image_scores: dict[str, float] = {}
    pixel_maps: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for images, names in loader:
            descriptors = extract_features(
                extractor,
                images.to(device),
                projection,
                CONFIG["descriptor_size"],
            )
            raw_maps = score_descriptors(descriptors, normal_bank)
            smoothed = smooth_maps(raw_maps, CONFIG["gaussian_sigma"])
            raw_image_scores = aggregate_maps(smoothed, CONFIG["top_fraction"])
            normalized_images = saturating_normalize(
                raw_image_scores,
                calibration["image_low"],
                calibration["image_scale"],
            )
            normalized_pixels = saturating_normalize(
                smoothed,
                calibration["pixel_low"],
                calibration["pixel_scale"],
            )
            normalized_pixels = F.interpolate(
                normalized_pixels[:, None],
                size=(CONFIG["image_size"], CONFIG["image_size"]),
                mode="bilinear",
                align_corners=False,
            )[:, 0].clamp(0, 1)

            for index, name in enumerate(names):
                image_scores[name] = float(normalized_images[index].cpu())
                pixel_maps[name] = normalized_pixels[index].cpu()
    return image_scores, pixel_maps


def main() -> None:
    if not CONFIG["model_path"].exists():
        raise RuntimeError(f"Model not found: run train.py first ({CONFIG['model_path']})")
    files = image_files(CONFIG["test_dir"])
    if not files:
        raise RuntimeError(f"No test images found in {CONFIG['test_dir']}")
    if len({path.name for path in files}) != len(files):
        raise RuntimeError("Test filenames must be unique for the required flat output format")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CONFIG["model_path"], map_location="cpu", weights_only=False)
    extractor = FeatureExtractor().eval().to(device)
    projection = checkpoint["projection"].to(device)
    normal_bank = checkpoint["normal_bank"]
    loader = DataLoader(
        ImageDataset(files, CONFIG["image_size"]),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Scoring {len(files)} test images on {device}...")
    image_scores, pixel_maps = score_images(
        extractor,
        loader,
        normal_bank,
        projection,
        checkpoint["calibration"],
        device,
    )

    pixel_dir = CONFIG["output_dir"] / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    for name, anomaly_map in pixel_maps.items():
        pixels = (anomaly_map.numpy() * 255).round().astype("uint8")
        Image.fromarray(pixels, mode="L").save(pixel_dir / name)
    with (CONFIG["output_dir"] / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(image_scores.items())), handle, indent=2)
        handle.write("\n")
    print(f"Wrote scores and {len(pixel_maps)} pixel maps to {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()

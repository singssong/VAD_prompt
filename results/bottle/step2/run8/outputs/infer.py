#!/usr/bin/env python3
"""Score test images and create dense anomaly maps."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from train import FeatureExtractor, ImageDataset, add_coordinates, image_files


def nearest_distances(
    queries: torch.Tensor, bank: torch.Tensor, chunk_size: int = 4096
) -> torch.Tensor:
    results = []
    bank_norm = (bank * bank).sum(dim=1).unsqueeze(0)
    for chunk in queries.split(chunk_size):
        distances = (
            (chunk * chunk).sum(dim=1, keepdim=True)
            + bank_norm
            - 2.0 * chunk @ bank.T
        )
        results.append(distances.clamp_min_(0).min(dim=1).values.sqrt_())
    return torch.cat(results)


def robust_image_score(anomaly_map: torch.Tensor) -> float:
    flat = anomaly_map.flatten()
    k = max(1, int(flat.numel() * 0.01))
    return float(torch.topk(flat, k).values.mean())


def foreground_mask(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    # White is the acquisition background. Expansion retains thin protruding defects.
    foreground = (rgb.min(axis=2) < 245).astype(np.uint8) * 255
    mask = Image.fromarray(foreground, mode="L").filter(ImageFilter.MaxFilter(15))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2.0))
    return torch.from_numpy(np.asarray(mask, dtype=np.float32).copy() / 255.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    files = image_files(args.test_dir)
    if not files:
        raise RuntimeError(f"No test images found in {args.test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)
    bank = checkpoint["memory_bank"].to(device=device, dtype=torch.float32)
    coordinate_weight = float(checkpoint["coordinate_weight"])
    model = FeatureExtractor().eval().to(device)
    loader = DataLoader(
        ImageDataset(files), batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == "cuda",
    )

    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    raw_maps: list[torch.Tensor] = []
    scores: dict[str, float] = {}
    offset = 0

    with torch.inference_mode():
        for images in loader:
            features = add_coordinates(
                model(images.to(device, non_blocking=True)), coordinate_weight
            )
            batch, channels, height, width = features.shape
            queries = features.permute(0, 2, 3, 1).reshape(-1, channels)
            distances = nearest_distances(queries, bank)
            maps = distances.reshape(batch, 1, height, width)
            maps = F.interpolate(
                maps, size=(256, 256), mode="bilinear", align_corners=False
            ).squeeze(1).cpu()
            for local_index, anomaly_map in enumerate(maps):
                path = files[offset + local_index]
                anomaly_map = anomaly_map * foreground_mask(path)
                raw_maps.append(anomaly_map)
                scores[path.name] = robust_image_score(anomaly_map)
            offset += batch

    # A common global scale preserves comparability between all output maps.
    all_values = torch.cat([m.flatten() for m in raw_maps])
    low = float(torch.quantile(all_values, 0.01))
    high = float(torch.quantile(all_values, 0.995))
    scale = max(high - low, 1e-8)
    for path, anomaly_map in zip(files, raw_maps):
        normalized = ((anomaly_map.numpy() - low) / scale * 255.0).clip(0, 255)
        image = Image.fromarray(normalized.astype(np.uint8), mode="L")
        image = image.filter(ImageFilter.GaussianBlur(radius=2.0))
        image.save(pixel_dir / path.name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(files)} images; outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()

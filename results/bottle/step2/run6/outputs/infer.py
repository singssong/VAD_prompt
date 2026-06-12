#!/usr/bin/env python3
"""Score test images with a trained PaDiM model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights
from torchvision.transforms import v2

from train import IMAGE_SUFFIXES, anomaly_maps, combine_features, make_extractor


class TestDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.paths = sorted(
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), self.paths[index].name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    transform = v2.Compose([
        v2.Resize((state["input_size"], state["input_size"]), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=Wide_ResNet50_2_Weights.DEFAULT.transforms().mean,
            std=Wide_ResNet50_2_Weights.DEFAULT.transforms().std,
        ),
    ])
    dataset = TestDataset(args.test_dir, transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == "cuda"
    )
    extractor = make_extractor(device)
    channel_indices = state["channel_indices"].to(device)
    mean = state["mean"].to(device)
    cholesky = state["cholesky"].to(device)
    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}
    top_k = max(1, int(0.01 * 256 * 256))

    with torch.inference_mode():
        for images, names in loader:
            outputs = extractor(images.to(device, non_blocking=True))
            features = combine_features(outputs, channel_indices)
            maps = anomaly_maps(features, mean, cholesky)
            raw_scores = maps.flatten(1).topk(top_k, dim=1).values.mean(dim=1)
            normalized_scores = (
                (raw_scores - state["image_center"]) / state["image_scale"]
            )
            png_maps = (
                (maps - state["pixel_low"])
                / max(state["pixel_high"] - state["pixel_low"], 1e-6)
            ).clamp(0, 1)
            png_maps = (png_maps.mul(255).round().byte().cpu().numpy())
            for name, score, pixel_map in zip(names, normalized_scores, png_maps):
                scores[name] = float(score.item())
                Image.fromarray(pixel_map[0], mode="L").save(pixel_dir / name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Scored {len(scores)} images; outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()

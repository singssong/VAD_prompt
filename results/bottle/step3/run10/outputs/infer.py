#!/usr/bin/env python3
"""Score test images against a learned normal patch-feature memory bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights

from train import IMAGE_EXTENSIONS, PatchFeatureExtractor


class TestDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.transform = ResNet50_Weights.IMAGENET1K_V2.transforms(
            crop_size=256, resize_size=256
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            return self.transform(image), path.name


def nearest_distances(
    features: torch.Tensor, memory_bank: torch.Tensor, chunk_size: int = 2048
) -> torch.Tensor:
    distances = []
    for chunk in features.split(chunk_size):
        max_similarity = torch.full(
            (len(chunk),), -1.0, dtype=torch.float32, device=chunk.device
        )
        for bank_chunk in memory_bank.split(8192):
            similarity = chunk @ bank_chunk.T
            max_similarity = torch.maximum(max_similarity, similarity.max(dim=1).values)
        distances.append(1.0 - max_similarity)
    return torch.cat(distances)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    paths = sorted(
        path for path in args.test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No test images found in {args.test_dir}")
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise RuntimeError("Test filenames must be unique")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    extractor = PatchFeatureExtractor().eval().to(device)
    projection = checkpoint["projection"].to(device)
    memory_bank = checkpoint["memory_bank"].float().to(device)
    memory_bank = F.normalize(memory_bank, dim=1)
    map_scale = float(checkpoint["map_scale"])

    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        TestDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    scores: dict[str, float] = {}
    with torch.inference_mode():
        for images, batch_names in loader:
            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]
            features = extractor(images)
            features = F.normalize(features @ projection, dim=1)
            patch_scores = nearest_distances(features, memory_bank)
            side = int(np.sqrt(patch_scores.numel() // batch_size))
            maps = patch_scores.reshape(batch_size, 1, side, side)
            maps = F.interpolate(maps, size=(256, 256), mode="bilinear", align_corners=False)
            maps = F.avg_pool2d(maps, kernel_size=9, stride=1, padding=4)

            for anomaly_map, name in zip(maps[:, 0], batch_names):
                flattened = anomaly_map.flatten()
                top_count = max(1, int(flattened.numel() * 0.01))
                image_score = float(flattened.topk(top_count).values.mean().cpu())
                scores[name] = image_score

                display_map = (anomaly_map / map_scale).clamp(0, 1)
                pixels = (display_map * 255).round().byte().cpu().numpy()
                Image.fromarray(pixels, mode="L").save(pixel_dir / name)

    ordered_scores = {name: scores[name] for name in sorted(scores)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(ordered_scores, handle, indent=2)
        handle.write("\n")
    print(f"Wrote scores and pixel maps for {len(scores)} test images")


if __name__ == "__main__":
    main()

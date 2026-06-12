#!/usr/bin/env python3
"""Score test images with a trained nearest-neighbor feature memory bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from train import (
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    FeatureExtractor,
    project_features,
)


class TestImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image, [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True
            )
            tensor = TF.to_tensor(image)
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD), path.name


def gaussian_kernel(size=9, sigma=2.0, device="cpu"):
    coordinates = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    return (kernel_1d[:, None] * kernel_1d[None, :])[None, None]


def nearest_neighbor_scores(queries, memory_bank, chunk_size=2048):
    scores = []
    memory_t = memory_bank.T.contiguous()
    for start in range(0, len(queries), chunk_size):
        similarities = queries[start : start + chunk_size] @ memory_t
        scores.append(1.0 - similarities.max(dim=1).values)
    return torch.cat(scores)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("./data/test_images"))
    parser.add_argument("--model-path", type=Path, default=Path("./outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)
    projection = checkpoint["projection"].to(device)
    memory_bank = F.normalize(checkpoint["memory_bank"].float(), dim=1).to(device)

    dataset = TestImageDataset(args.test_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, (len(dataset) + args.batch_size - 1) // args.batch_size),
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().eval().to(device)
    kernel = gaussian_kernel(device=device)

    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    image_scores = {}

    with torch.inference_mode():
        for batch_index, (images, names) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            features = extractor(images)
            patches = project_features(features, projection)
            patch_scores = nearest_neighbor_scores(patches, memory_bank)
            maps = patch_scores.reshape(len(images), 1, 32, 32)
            maps = F.interpolate(maps, size=(256, 256), mode="bilinear", align_corners=False)
            maps = F.conv2d(maps, kernel, padding=kernel.shape[-1] // 2)

            for anomaly_map, name in zip(maps[:, 0], names):
                flat = anomaly_map.flatten()
                top_count = max(1, int(0.01 * flat.numel()))
                image_scores[name] = float(torch.topk(flat, top_count).values.mean().item())

                map_array = anomaly_map.cpu().numpy()
                low, high = np.percentile(map_array, [1.0, 99.0])
                if high > low:
                    map_array = np.clip((map_array - low) / (high - low), 0.0, 1.0)
                else:
                    map_array = np.zeros_like(map_array)
                Image.fromarray(np.round(map_array * 255).astype(np.uint8), mode="L").save(
                    pixel_dir / name
                )
            print(f"Scored batch {batch_index}/{len(loader)}", flush=True)

    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as output:
        json.dump(image_scores, output, indent=2, sort_keys=True)
        output.write("\n")
    print(f"Wrote scores for {len(image_scores)} images to {args.output_dir}")


if __name__ == "__main__":
    main()

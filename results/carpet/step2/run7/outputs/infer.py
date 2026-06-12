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

from train import FeatureExtractor, IMAGE_EXTENSIONS, MEAN, STD, image_files


class TestDataset(Dataset):
    def __init__(self, directory: Path):
        self.paths = image_files(directory)
        if not self.paths:
            raise RuntimeError(f"No images found in {directory}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
        mean = torch.tensor(MEAN)[:, None, None]
        std = torch.tensor(STD)[:, None, None]
        return (tensor - mean) / std, path.name


def nearest_distances(queries, memory, chunk_size=4096):
    results = []
    memory_t = memory.T.contiguous()
    memory_norm = (memory * memory).sum(dim=1)
    for start in range(0, len(queries), chunk_size):
        query = queries[start:start + chunk_size]
        squared = (
            (query * query).sum(dim=1, keepdim=True)
            + memory_norm[None, :]
            - 2.0 * (query @ memory_t)
        )
        results.append(squared.clamp_min_(0).min(dim=1).values.sqrt_())
    return torch.cat(results)


def gaussian_kernel(size=21, sigma=4.0, device="cpu"):
    coords = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return (kernel[:, None] @ kernel[None, :])[None, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--memory-chunk", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    memory = checkpoint["memory_bank"].float().to(device)
    projection = checkpoint["projection"].float().to(device)
    grid_size = int(checkpoint["feature_grid"])

    dataset = TestDataset(args.test_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().to(device)
    kernel = gaussian_kernel(device=device)
    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}

    with torch.inference_mode():
        for images, names in loader:
            features = extractor(images.to(device, non_blocking=True))
            batch_size, channels, height, width = features.shape
            queries = features.permute(0, 2, 3, 1).reshape(-1, channels)
            queries = F.normalize(queries @ projection, dim=1)
            distances = nearest_distances(queries, memory, args.memory_chunk)
            maps = distances.reshape(batch_size, grid_size, grid_size)[:, None]
            maps = F.interpolate(maps, size=(256, 256), mode="bilinear", align_corners=False)
            maps = F.conv2d(maps, kernel, padding=kernel.shape[-1] // 2)

            for anomaly_map, name in zip(maps[:, 0], names):
                flat = anomaly_map.flatten()
                top_count = max(1, int(0.01 * flat.numel()))
                image_score = flat.topk(top_count).values.mean().item()
                scores[name] = float(image_score)

                array = anomaly_map.cpu().numpy()
                low, high = np.percentile(array, (1, 99))
                scaled = np.clip((array - low) / max(high - low, 1e-8), 0, 1)
                Image.fromarray(np.round(scaled * 255).astype(np.uint8), mode="L").save(
                    pixel_dir / name
                )

    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images using {device}.")


if __name__ == "__main__":
    main()

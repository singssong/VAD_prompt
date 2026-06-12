#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights

from train import FeatureExtractor, IMAGE_EXTENSIONS


class TestDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(
            p for p in Path(root).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.files:
            raise RuntimeError(f"No images found in {root}")
        self.transform = Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms(
            crop_size=256, resize_size=256, antialias=True
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.files[index].name


def gaussian_kernel(size=9, sigma=2.0):
    coordinates = torch.arange(size, dtype=torch.float32) - size // 2
    kernel = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return (kernel[:, None] * kernel[None, :])[None, None]


def nearest_cosine_distance(patches, memory_bank, memory_chunk=5000):
    best_similarity = torch.full(
        (len(patches),), -1.0, device=patches.device, dtype=patches.dtype
    )
    for start in range(0, len(memory_bank), memory_chunk):
        similarities = patches @ memory_bank[start:start + memory_chunk].T
        best_similarity = torch.maximum(best_similarity, similarities.max(dim=1).values)
    return (1.0 - best_similarity).clamp_min_(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="./data/test_images")
    parser.add_argument("--model", default="./outputs/model.pt")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    dataset = TestDataset(args.test_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(4, max(0, len(dataset) // 32)), pin_memory=device.type == "cuda"
    )

    extractor = FeatureExtractor().to(device)
    indices = state["channel_indices"].to(device)
    memory_bank = state["memory_bank"].to(device=device, dtype=torch.float32)
    smoothing = gaussian_kernel().to(device)
    # Cosine distance 0.08 is a stable visualization ceiling for these features.
    map_scale = 0.08

    output_dir = Path(args.output_dir)
    pixel_dir = output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}

    with torch.inference_mode():
        for images, names in loader:
            features = extractor(images.to(device, non_blocking=True)).index_select(1, indices)
            batch_size, channels, height, width = features.shape
            patches = features.permute(0, 2, 3, 1).reshape(-1, channels)
            patches = F.normalize(patches, dim=1)
            distances = nearest_cosine_distance(patches, memory_bank)
            maps = distances.reshape(batch_size, 1, height, width)
            maps = F.interpolate(maps, size=(256, 256), mode="bilinear", align_corners=False)
            maps = F.conv2d(maps, smoothing, padding=smoothing.shape[-1] // 2)

            flat = maps.flatten(1)
            top_count = max(1, int(flat.shape[1] * 0.01))
            image_scores = flat.topk(top_count, dim=1).values.mean(dim=1)
            png_maps = (maps[:, 0] / map_scale * 255.0).clamp(0, 255).byte().cpu().numpy()

            for name, score, pixel_map in zip(names, image_scores.cpu().tolist(), png_maps):
                scores[name] = float(score)
                Image.fromarray(pixel_map, mode="L").save(pixel_dir / name)

    expected = {p.name for p in dataset.files}
    if set(scores) != expected:
        raise RuntimeError("Inference did not produce a score for every test image")
    with (output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {len(scores)} image scores and pixel maps to {output_dir}.")


if __name__ == "__main__":
    main()

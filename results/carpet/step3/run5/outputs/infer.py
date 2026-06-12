#!/usr/bin/env python3
"""Score test images against a normal-patch feature memory bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, pil_to_tensor, resize

from train import IMAGE_EXTENSIONS, FeatureExtractor


class TestDataset(Dataset):
    def __init__(self, root: Path):
        self.files = sorted(
            p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.files:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = resize(image, [256, 256], interpolation=InterpolationMode.BILINEAR)
            tensor = pil_to_tensor(image).float().div_(255.0)
        tensor = normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return tensor, path.name


def nearest_cosine_distance(features, memory_bank, chunk_size=512):
    """Return 1 - maximum cosine similarity for each patch."""
    output = []
    for start in range(0, len(features), chunk_size):
        query = features[start : start + chunk_size]
        max_similarity = torch.matmul(query, memory_bank.T).amax(dim=1)
        output.append(1.0 - max_similarity.float())
    return torch.cat(output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nn-chunk-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    memory_bank = checkpoint["memory_bank"].to(device)
    if device.type == "cpu":
        memory_bank = memory_bank.float()

    dataset = TestDataset(args.test_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().to(device)
    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}

    with torch.inference_mode():
        completed = 0
        for images, names in loader:
            feature_maps = extractor(images.to(device, non_blocking=True))
            for feature_map, name in zip(feature_maps, names):
                patches = feature_map.permute(1, 2, 0).reshape(-1, feature_map.shape[0])
                patches = F.normalize(patches, dim=1).to(memory_bank.dtype)
                distances = nearest_cosine_distance(
                    patches, memory_bank, chunk_size=args.nn_chunk_size
                )
                anomaly_map = distances.reshape(1, 1, *feature_map.shape[-2:])
                anomaly_map = F.interpolate(
                    anomaly_map, size=(256, 256), mode="bilinear", align_corners=False
                )
                anomaly_map = F.avg_pool2d(anomaly_map, kernel_size=9, stride=1, padding=4)
                anomaly_map = anomaly_map.squeeze().clamp_min(0)

                flat = anomaly_map.flatten()
                tail_count = max(1, int(flat.numel() * 0.01))
                image_score = flat.topk(tail_count).values.mean().item()
                scores[name] = float(image_score)

                # Cosine distance has a fixed theoretical range [0, 2].
                png_values = (
                    anomaly_map.clamp(0, 2).div(2).mul(65535).round().cpu().numpy().astype(np.uint16)
                )
                Image.fromarray(png_values).save(pixel_dir / name)
                completed += 1
                print(f"\rScored {completed}/{len(dataset)}", end="")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"\nSaved scores and maps to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score test images using nearest-neighbor distance to normal patches."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights

from train import FeatureExtractor, IMAGE_EXTENSIONS


class TestDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


def gaussian_kernel(size=17, sigma=4.0):
    coordinates = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    kernel = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :])[None, None]


def nearest_distances(queries, memory_bank, chunk_size):
    results = []
    memory_t = memory_bank.T.contiguous()
    for start in range(0, len(queries), chunk_size):
        chunk = queries[start : start + chunk_size]
        # Unit-normalized vectors: squared L2 distance = 2 - 2*cosine similarity.
        similarities = chunk @ memory_t
        results.append(torch.sqrt(torch.clamp(2.0 - 2.0 * similarities.max(dim=1).values, min=0.0)))
    return torch.cat(results)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=script_dir.parent / "data" / "test_images")
    parser.add_argument("--model-path", type=Path, default=script_dir / "patchcore_model.pt")
    parser.add_argument("--scores-path", type=Path, default=script_dir / "image_scores.json")
    parser.add_argument("--pixel-dir", type=Path, default=script_dir / "pixel_scores")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--distance-chunk-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)
    memory_bank = F.normalize(checkpoint["memory_bank"].float(), dim=1).to(args.device)
    extractor = FeatureExtractor().to(args.device).eval()
    dataset = TestDataset(args.test_dir, ResNet18_Weights.IMAGENET1K_V1.transforms())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, len(dataset)),
        pin_memory=args.device.startswith("cuda"),
    )
    blur_kernel = gaussian_kernel().to(args.device)
    args.pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}

    with torch.inference_mode():
        for images, names in loader:
            features = extractor(images.to(args.device, non_blocking=True))
            batch_size, channels, height, width = features.shape
            patches = features.permute(0, 2, 3, 1).reshape(-1, channels)
            patches = F.normalize(patches, dim=1)
            distances = nearest_distances(patches, memory_bank, args.distance_chunk_size)
            maps = distances.reshape(batch_size, 1, height, width)
            maps = F.interpolate(maps, size=(256, 256), mode="bilinear", align_corners=False)
            maps = F.conv2d(maps, blur_kernel, padding=blur_kernel.shape[-1] // 2)

            for anomaly_map, name in zip(maps[:, 0], names):
                flat = anomaly_map.flatten()
                top_count = max(1, math.ceil(flat.numel() * 0.01))
                image_score = flat.topk(top_count).values.mean().item()
                if not math.isfinite(image_score):
                    raise RuntimeError(f"Non-finite score for {name}")
                scores[name] = float(image_score)

                # Use one fixed encoding for every image so PNG values remain
                # comparable across the full test set.
                scaled = anomaly_map.clamp(0, 1)
                output = Image.fromarray((scaled * 255).round().byte().cpu().numpy(), mode="L")
                output.save(args.pixel_dir / name)

    args.scores_path.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_path.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Scored {len(scores)} images; wrote {args.scores_path} and {args.pixel_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score test images and write image-level and pixel-level anomaly scores."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights

from train import FeatureExtractor, image_files


class TestDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.files = image_files(root)
        if not self.files:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB")), self.files[index].name


def gaussian_kernel(size=21, sigma=4.0):
    axis = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    kernel = torch.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * sigma**2))
    return (kernel / kernel.sum())[None, None]


def nearest_distances(queries, bank, chunk_size=4096):
    results = []
    bank_t = bank.T.contiguous()
    for chunk in queries.split(chunk_size):
        # Both vectors are unit length, so squared L2 distance is 2 - 2*cosine.
        similarity = chunk @ bank_t
        results.append((2.0 - 2.0 * similarity.max(dim=1).values).clamp_min_(0).sqrt_())
    return torch.cat(results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    projection = checkpoint["projection"].to(device)
    memory_bank = checkpoint["memory_bank"].float().to(device)

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    dataset = TestDataset(args.test_dir, weights.transforms())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, len(dataset)),
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().eval().to(device)
    kernel = gaussian_kernel().to(device)
    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}
    anomaly_maps = {}

    with torch.inference_mode():
        for images, names in loader:
            features = extractor(images.to(device, non_blocking=True))
            batch, channels, height, width = features.shape
            patches = features.permute(0, 2, 3, 1).reshape(-1, channels)
            patches = F.normalize(patches @ projection, dim=1)
            distances = nearest_distances(patches, memory_bank).view(batch, 1, height, width)
            maps = F.interpolate(distances, size=(256, 256), mode="bilinear", align_corners=False)
            padding = kernel.shape[-1] // 2
            maps = F.conv2d(F.pad(maps, (padding,) * 4, mode="reflect"), kernel)

            for anomaly_map, name in zip(maps[:, 0], names):
                # A top-tail mean is robust to isolated noise while rewarding defect extent.
                flat = anomaly_map.flatten()
                top_count = max(1, flat.numel() // 100)
                image_score = flat.topk(top_count).values.mean().item()
                scores[name] = float(image_score)
                anomaly_maps[name] = anomaly_map.cpu()

    # A shared calibration preserves pixel-score ordering between test images.
    all_pixels = torch.cat([anomaly_map.flatten() for anomaly_map in anomaly_maps.values()])
    low, high = torch.quantile(all_pixels, torch.tensor([0.01, 0.995]))
    for name, anomaly_map in anomaly_maps.items():
        normalized = ((anomaly_map - low) / (high - low + 1e-8)).clamp(0, 1)
        output = Image.fromarray(normalized.mul(255).byte().numpy(), mode="L")
        output.save(pixel_dir / name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images on {device}; outputs are in {args.output_dir}.")


if __name__ == "__main__":
    main()

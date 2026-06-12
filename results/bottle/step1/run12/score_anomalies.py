#!/usr/bin/env python3
"""Score test images with a training-only PatchCore-style memory bank."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.paths = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
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


class PatchEmbedder:
    def __init__(self, device: torch.device, projection_dim: int, seed: int):
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        self.model = wide_resnet50_2(weights=weights).to(device).eval()
        self.device = device
        self.activations = {}
        self.model.layer2.register_forward_hook(self._save("layer2"))
        self.model.layer3.register_forward_hook(self._save("layer3"))

        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(1536, projection_dim, generator=generator)
        self.projection = (projection / projection_dim**0.5).to(device)

    def _save(self, name):
        def hook(_module, _inputs, output):
            self.activations[name] = output
        return hook

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        self.activations.clear()
        self.model(images.to(self.device, non_blocking=True))

        layer2 = F.avg_pool2d(self.activations["layer2"], 3, stride=1, padding=1)
        layer3 = F.avg_pool2d(self.activations["layer3"], 3, stride=1, padding=1)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )

        # Equalize the contribution of each backbone level before concatenation.
        layer2 = F.normalize(layer2, dim=1)
        layer3 = F.normalize(layer3, dim=1)
        features = torch.cat((layer2, layer3), dim=1)
        features = features.permute(0, 2, 3, 1).reshape(features.shape[0], -1, 1536)
        return features @ self.projection


def build_memory_bank(embedder, loader, max_patches: int, seed: int):
    batches = []
    for images, _names in loader:
        batches.append(embedder(images).cpu())
    patches = torch.cat(batches, dim=0).reshape(-1, batches[0].shape[-1])

    if len(patches) > max_patches:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(patches), generator=generator)[:max_patches]
        patches = patches[indices]
    return patches.contiguous()


@torch.inference_mode()
def score_batch(features, memory_bank, chunk_size=10_000):
    batch_size, patch_count, feature_dim = features.shape
    flat = features.reshape(-1, feature_dim)
    min_distances = torch.full(
        (len(flat),), float("inf"), device=flat.device, dtype=flat.dtype
    )
    flat_norm = (flat * flat).sum(dim=1, keepdim=True)

    for start in range(0, len(memory_bank), chunk_size):
        bank = memory_bank[start : start + chunk_size]
        bank_norm = (bank * bank).sum(dim=1).unsqueeze(0)
        squared = flat_norm + bank_norm - 2.0 * (flat @ bank.T)
        min_distances = torch.minimum(min_distances, squared.min(dim=1).values)

    patch_scores = min_distances.clamp_min_(0).sqrt_().reshape(batch_size, patch_count)
    top_k = max(1, int(round(patch_count * 0.01)))
    return patch_scores.topk(top_k, dim=1).values.mean(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-patches", type=int, default=40_000)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    transform = weights.transforms()
    train_dataset = ImageDataset(args.train_dir, transform)
    test_dataset = ImageDataset(args.test_dir, transform)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    embedder = PatchEmbedder(device, args.projection_dim, args.seed)
    memory_bank = build_memory_bank(
        embedder, train_loader, args.memory_patches, args.seed
    ).to(device)

    rows = []
    for images, names in test_loader:
        features = embedder(images)
        scores = score_batch(features, memory_bank)
        rows.extend(zip(names, scores.cpu().tolist()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "anomaly_score"])
        writer.writerows((name, f"{score:.10f}") for name, score in rows)

    metadata = {
        "method": "PatchCore-style patch nearest-neighbor anomaly detection",
        "backbone": "ImageNet-pretrained Wide ResNet-50-2",
        "train_images": len(train_dataset),
        "test_images": len(test_dataset),
        "memory_patches": len(memory_bank),
        "projection_dim": args.projection_dim,
        "score_aggregation": "mean of top 1% patch distances",
        "output": str(args.output),
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

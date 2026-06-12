#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from common import FeatureExtractor, ImageDataset, flatten_features


def parse_args():
    parser = argparse.ArgumentParser(description="Train a normal patch prototype model.")
    parser.add_argument("--train-dir", default="./data/train")
    parser.add_argument("--output", default="./outputs/model.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prototypes", type=int, default=256)
    parser.add_argument("--max-patches", type=int, default=100000)
    parser.add_argument("--kmeans-iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.inference_mode()
def extract_patch_sample(model, loader, device, max_patches, seed):
    chunks = []
    for images, _ in loader:
        chunks.append(flatten_features(model(images.to(device))).cpu())
    patches = torch.cat(chunks)
    generator = torch.Generator().manual_seed(seed)
    if len(patches) > max_patches:
        indices = torch.randperm(len(patches), generator=generator)[:max_patches]
        patches = patches[indices]
    return patches


@torch.inference_mode()
def spherical_kmeans(patches, count, iterations, device, seed):
    generator = torch.Generator().manual_seed(seed)
    initial = torch.randperm(len(patches), generator=generator)[:count]
    centers = patches[initial].to(device)
    centers = torch.nn.functional.normalize(centers, dim=1)
    patches = patches.to(device)

    for iteration in range(iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(count, device=device)
        objective = 0.0
        for batch in patches.split(8192):
            similarities = batch @ centers.T
            values, assignments = similarities.max(dim=1)
            sums.index_add_(0, assignments, batch)
            counts.index_add_(0, assignments, torch.ones_like(values))
            objective += values.sum().item()
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]
        centers = torch.nn.functional.normalize(centers, dim=1)
        print(
            f"k-means {iteration + 1}/{iterations}: "
            f"mean cosine similarity={objective / len(patches):.6f}"
        )
    return centers


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    extractor = FeatureExtractor()
    pretrained = resnet18(weights=ResNet18_Weights.DEFAULT)
    extractor.load_state_dict(
        {
            key: value
            for key, value in pretrained.state_dict().items()
            if key in extractor.state_dict()
        },
        strict=True,
    )
    extractor.eval().requires_grad_(False).to(device)

    patches = extract_patch_sample(
        extractor, loader, device, args.max_patches, args.seed
    )
    if args.prototypes > len(patches):
        raise ValueError("Number of prototypes exceeds number of sampled patches")
    print(f"clustering {len(patches)} patches into {args.prototypes} prototypes")
    prototypes = spherical_kmeans(
        patches, args.prototypes, args.kmeans_iterations, device, args.seed
    )

    # Derive one fixed visualization scale from normal training distances.
    calibration = []
    for batch in patches.to(device).split(8192):
        calibration.append((1.0 - (batch @ prototypes.T).max(dim=1).values).cpu())
    calibration = torch.cat(calibration)
    low = torch.quantile(calibration, 0.01).item()
    high = torch.quantile(calibration, 0.995).item()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "extractor": extractor.cpu().state_dict(),
            "prototypes": prototypes.cpu(),
            "calibration_low": low,
            "calibration_high": high,
            "backbone": "resnet18",
            "feature_layers": ["layer1", "layer2"],
            "input_size": 256,
        },
        output,
    )
    print(f"saved {output} (normal distance range {low:.6f} to {high:.6f})")


if __name__ == "__main__":
    main()


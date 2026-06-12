#!/usr/bin/env python3
"""Unsupervised image anomaly scoring from a normal reference directory."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


def image_paths(directory: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions)


class FeatureExtractor:
    def __init__(self, device: torch.device) -> None:
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        self.model = wide_resnet50_2(weights=weights).to(device).eval()
        self.preprocess = weights.transforms()
        self.device = device
        self.features: dict[str, torch.Tensor] = {}
        self.model.layer2.register_forward_hook(self._capture("layer2"))
        self.model.layer3.register_forward_hook(self._capture("layer3"))

    def _capture(self, name: str):
        def hook(_module, _inputs, output):
            self.features[name] = output

        return hook

    @torch.inference_mode()
    def __call__(self, paths: list[Path], batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        all_patches, all_global = [], []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = torch.stack(
                [self.preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
            ).to(self.device)
            self.model(images)

            layer2 = F.avg_pool2d(self.features["layer2"], 3, stride=1, padding=1)
            layer3 = F.avg_pool2d(self.features["layer3"], 3, stride=1, padding=1)
            layer3 = F.interpolate(layer3, layer2.shape[-2:], mode="bilinear", align_corners=False)
            patches = torch.cat((layer2, layer3), dim=1)
            patches = F.normalize(patches, dim=1)
            patches = patches.permute(0, 2, 3, 1).flatten(1, 2).cpu()

            global_features = F.normalize(self.features["layer3"].mean((2, 3)), dim=1).cpu()
            all_patches.append(patches)
            all_global.append(global_features)
        return torch.cat(all_patches), torch.cat(all_global)


def make_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    return projection / np.sqrt(output_dim)


def project(features: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    return F.normalize(features @ projection, dim=-1)


@torch.inference_mode()
def local_scores(
    test_patches: torch.Tensor,
    memory: torch.Tensor,
    device: torch.device,
    image_chunk: int = 4,
    memory_chunk: int = 4096,
) -> torch.Tensor:
    scores = []
    memory = memory.to(device)
    for start in range(0, len(test_patches), image_chunk):
        patches = test_patches[start : start + image_chunk].to(device)
        nearest = torch.full(patches.shape[:2], float("inf"), device=device)
        for mem_start in range(0, len(memory), memory_chunk):
            mem = memory[mem_start : mem_start + memory_chunk]
            # Unit vectors: squared Euclidean distance is 2 - 2 cosine similarity.
            distances = 2.0 - 2.0 * torch.matmul(patches, mem.T)
            nearest = torch.minimum(nearest, distances.min(dim=-1).values)
        top_k = max(1, nearest.shape[1] // 100)
        scores.append(nearest.topk(top_k, dim=1).values.mean(dim=1).cpu())
    return torch.cat(scores)


def robust_standardize(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    median = reference.median()
    mad = (reference - median).abs().median().clamp_min(1e-6)
    return (values - median) / (1.4826 * mad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--memory-size", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths = image_paths(args.train_dir)
    test_paths = image_paths(args.test_dir)
    if not train_paths or not test_paths:
        raise RuntimeError("Both train and test directories must contain images")

    extractor = FeatureExtractor(device)
    train_patches, train_global = extractor(train_paths, args.batch_size)
    test_patches, test_global = extractor(test_paths, args.batch_size)

    projection = make_projection(train_patches.shape[-1], 256, args.seed)
    train_patches = project(train_patches, projection)
    test_patches = project(test_patches, projection)

    calibration_count = min(48, max(1, len(train_paths) // 4))
    flat_train = train_patches[calibration_count:].flatten(0, 1)
    generator = torch.Generator().manual_seed(args.seed)
    memory_indices = torch.randperm(len(flat_train), generator=generator)[: args.memory_size]
    memory = flat_train[memory_indices]

    test_local = local_scores(test_patches, memory, device)
    # Reference calibration uses a disjoint memory split to avoid self-neighbor distances.
    calibration_local = local_scores(train_patches[:calibration_count], memory, device)

    global_center = train_global.mean(dim=0)
    test_global_distance = 1.0 - F.cosine_similarity(test_global, global_center[None])
    train_global_distance = 1.0 - F.cosine_similarity(train_global, global_center[None])

    local_z = robust_standardize(test_local, calibration_local)
    global_z = robust_standardize(test_global_distance, train_global_distance)
    combined = local_z + 0.35 * global_z

    # Preserve the detector's full ranking while returning portable [0, 1] scores.
    score_range = (combined.max() - combined.min()).clamp_min(1e-12)
    scores = ((combined - combined.min()) / score_range).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "anomaly_score"])
        writer.writerows((path.name, f"{score:.10f}") for path, score in zip(test_paths, scores))

    print(f"Wrote {len(scores)} scores to {args.output}")
    print("Method: PatchCore-style local nearest-neighbor scoring + global feature distance")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a one-class PatchCore-style memory bank from normal images."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_batch(paths: list[Path], device: torch.device) -> torch.Tensor:
    arrays = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            arrays.append(np.asarray(image, dtype=np.float32) / 255.0)
    batch = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return ((batch - mean) / std).to(device)


class FeatureExtractor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool, model.layer1
        )
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        raw_level2 = self.layer2(x)
        raw_level3 = self.layer3(raw_level2)
        level2 = F.avg_pool2d(raw_level2, 3, stride=1, padding=1)
        level3 = F.avg_pool2d(raw_level3, 3, stride=1, padding=1)
        level3 = F.interpolate(
            level3, size=level2.shape[-2:], mode="bilinear", align_corners=False
        )
        level2 = F.normalize(level2, dim=1)
        level3 = F.normalize(level3, dim=1)
        features = torch.cat((level2, level3), dim=1)
        return features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])


@torch.inference_mode()
def descriptors(
    extractor: FeatureExtractor,
    paths: list[Path],
    projection: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    output = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        raw = extractor(load_batch(batch_paths, device))
        projected = F.normalize(raw @ projection, dim=1)
        side = int(round((projected.shape[0] / len(batch_paths)) ** 0.5))
        output.extend(projected.reshape(len(batch_paths), side * side, -1).cpu())
    return output


@torch.inference_mode()
def nearest_distances(
    queries: torch.Tensor, memory: torch.Tensor, device: torch.device
) -> torch.Tensor:
    memory = memory.to(device)
    result = []
    for start in range(0, len(queries), 4096):
        query = queries[start:start + 4096].to(device)
        best_similarity = torch.full((len(query),), -1.0, device=device)
        for memory_start in range(0, len(memory), 8192):
            similarities = query @ memory[memory_start:memory_start + 8192].T
            best_similarity = torch.maximum(best_similarity, similarities.max(dim=1).values)
        result.append(torch.sqrt((2.0 - 2.0 * best_similarity).clamp_min(0)).cpu())
    return torch.cat(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-size", type=int, default=30000)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = image_files(args.train_dir)
    if not paths:
        raise RuntimeError(f"No training images found in {args.train_dir}")
    shuffled = paths.copy()
    random.shuffle(shuffled)
    calibration_count = min(40, max(1, len(paths) // 7))
    calibration_paths = shuffled[:calibration_count]
    memory_paths = shuffled[calibration_count:]

    extractor = FeatureExtractor().to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    projection = torch.randn(
        1536, args.projection_dim, generator=generator, device=device
    ) / np.sqrt(args.projection_dim)

    print(f"Extracting normal features from {len(memory_paths)} images on {device}...")
    memory_per_image = descriptors(
        extractor, memory_paths, projection, args.batch_size, device
    )
    all_memory = torch.cat(memory_per_image)
    sample_count = min(args.memory_size, len(all_memory))
    indices = torch.randperm(len(all_memory), generator=torch.Generator().manual_seed(args.seed))
    memory = all_memory[indices[:sample_count]].contiguous()

    print(f"Calibrating on {len(calibration_paths)} held-out normal images...")
    calibration = descriptors(
        extractor, calibration_paths, projection, args.batch_size, device
    )
    calibration_distances = nearest_distances(torch.cat(calibration), memory, device)
    median = calibration_distances.median()
    mad = (calibration_distances - median).abs().median()
    scale = max(float(1.4826 * mad), 1e-4)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory": memory.half(),
            "projection": projection.cpu().half(),
            "calibration_median": float(median),
            "calibration_scale": scale,
            "input_size": 256,
            "backbone": "wide_resnet50_2",
            "seed": args.seed,
        },
        args.model_out,
    )
    print(
        f"Saved {sample_count} normal patch descriptors to {args.model_out} "
        f"(median={float(median):.4f}, scale={scale:.4f})"
    )


if __name__ == "__main__":
    main()

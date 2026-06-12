#!/usr/bin/env python3
"""Train a one-class patch-memory anomaly detector on normal images."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image, [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True
            )
            tensor = TF.to_tensor(image)
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class ResNet18Features(nn.Module):
    """Frozen ImageNet ResNet-18 features at 1/4, 1/8, and 1/16 resolution."""

    def __init__(self):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        x1 = self.layer1(self.stem(images))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x1 = F.adaptive_avg_pool2d(x1, x2.shape[-2:])
        x3 = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((x1, x2, x3), dim=1)


def project_patches(features, projection):
    patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    patches = patches @ projection
    return F.normalize(patches, dim=1)


def nearest_distances(queries, memory, chunk_size=1024):
    results = []
    memory_t = memory.T.contiguous()
    for chunk in queries.split(chunk_size):
        max_similarity = chunk @ memory_t
        max_similarity = max_similarity.max(dim=1).values.clamp(-1.0, 1.0)
        results.append(torch.sqrt((2.0 - 2.0 * max_similarity).clamp_min(0.0)))
    return torch.cat(results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("./data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("./outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--memory-size", type=int, default=5000)
    parser.add_argument("--calibration-size", type=int, default=5000)
    parser.add_argument("--projection-dim", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, len(dataset)),
        pin_memory=device.type == "cuda",
    )
    extractor = ResNet18Features().to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    projection = torch.randn(448, args.projection_dim, generator=generator, device=device)
    projection = projection / np.sqrt(args.projection_dim)

    all_patches = []
    with torch.inference_mode():
        for images in loader:
            features = extractor(images.to(device, non_blocking=True))
            all_patches.append(project_patches(features, projection).cpu())
    all_patches = torch.cat(all_patches)

    cpu_generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(all_patches), generator=cpu_generator)
    memory_count = min(args.memory_size, len(order))
    calibration_count = min(args.calibration_size, len(order) - memory_count)
    if calibration_count == 0:
        raise RuntimeError("Not enough training patches for separate calibration patches")

    memory = all_patches[order[:memory_count]].to(device)
    calibration = all_patches[
        order[memory_count : memory_count + calibration_count]
    ].to(device)
    with torch.inference_mode():
        calibration_distances = nearest_distances(calibration, memory)
    low = float(torch.quantile(calibration_distances, 0.50).cpu())
    high = float(torch.quantile(calibration_distances, 0.995).cpu())
    if high <= low:
        high = low + 1e-6

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": "resnet18",
            "weights": "IMAGENET1K_V1",
            "input_size": 256,
            "feature_map_size": 32,
            "projection": projection.cpu(),
            "memory_bank": memory.cpu(),
            "calibration_low": low,
            "calibration_high": high,
            "seed": args.seed,
        },
        args.model_out,
    )
    print(
        f"Saved {args.model_out}: {len(dataset)} images, "
        f"{memory_count} memory patches, calibration=[{low:.6f}, {high:.6f}]"
    )


if __name__ == "__main__":
    main()

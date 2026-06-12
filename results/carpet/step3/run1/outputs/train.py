#!/usr/bin/env python3
"""Train a feature-memory anomaly detector using only normal images."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import v2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_files(directory: Path) -> list[Path]:
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise RuntimeError(f"No supported images found in {directory}")
    return files


class ImageDataset(Dataset):
    def __init__(self, directory: Path):
        self.paths = image_files(directory)
        self.transform = v2.Compose([
            v2.Resize((256, 256), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


class PatchFeatureExtractor(nn.Module):
    """ImageNet WRN-50-2 features fused onto a 16x16 patch grid."""

    def __init__(self):
        super().__init__()
        self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        x = model.conv1(images)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        layer2 = model.layer2(x)
        layer3 = model.layer3(layer2)

        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer2 = F.adaptive_avg_pool2d(layer2, (16, 16))
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        fused = torch.cat((layer2, layer3), dim=1)
        return fused.permute(0, 2, 3, 1).reshape(images.shape[0], 256, -1)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--memory-size", type=int, default=30000)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    extractor = PatchFeatureExtractor().to(device)

    generator = torch.Generator().manual_seed(args.seed)
    channel_index = torch.randperm(1536, generator=generator)[:args.embedding_dim]
    embeddings = []
    with torch.inference_mode():
        for images, _ in loader:
            features = extractor(images.to(device, non_blocking=True))
            embeddings.append(features[:, :, channel_index.to(device)].cpu())

    patches = torch.cat(embeddings, dim=0).reshape(-1, args.embedding_dim)
    feature_mean = patches.mean(dim=0)
    feature_std = patches.std(dim=0).clamp_min(1e-4)
    patches = (patches - feature_mean) / feature_std

    memory_size = min(args.memory_size, patches.shape[0])
    memory_index = torch.randperm(patches.shape[0], generator=generator)[:memory_size]
    memory_bank = patches[memory_index].contiguous()

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "backbone": "wide_resnet50_2",
        "weights": "IMAGENET1K_V2",
        "input_size": 256,
        "grid_size": 16,
        "channel_index": channel_index,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "memory_bank": memory_bank.to(torch.float16),
        "train_image_count": len(dataset),
        "seed": args.seed,
    }, args.model_out)
    print(
        f"Saved {memory_size} normal patch features from {len(dataset)} images "
        f"to {args.model_out} using {device}."
    )


if __name__ == "__main__":
    main()

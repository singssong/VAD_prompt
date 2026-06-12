#!/usr/bin/env python3
"""Build a PatchCore-style memory bank from normal training images."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def image_files(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, directory: Path):
        self.paths = image_files(directory)
        if not self.paths:
            raise RuntimeError(f"No images found in {directory}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            array = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
        mean = torch.tensor(MEAN)[:, None, None]
        std = torch.tensor(STD)[:, None, None]
        return (tensor - mean) / std


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((layer2, layer3), dim=1)


def main():
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

    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().to(device)

    feature_dim = 1536
    generator = torch.Generator().manual_seed(args.seed)
    projection = torch.randn(feature_dim, args.projection_dim, generator=generator)
    projection /= torch.linalg.vector_norm(projection, dim=0, keepdim=True)
    projection = projection.to(device)

    patches = []
    with torch.inference_mode():
        for images in loader:
            features = extractor(images.to(device, non_blocking=True))
            features = features.permute(0, 2, 3, 1).reshape(-1, feature_dim)
            features = F.normalize(features @ projection, dim=1)
            patches.append(features.cpu())

    all_patches = torch.cat(patches)
    if len(all_patches) > args.memory_size:
        indices = torch.randperm(len(all_patches), generator=generator)[:args.memory_size]
        memory_bank = all_patches[indices]
    else:
        memory_bank = all_patches

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": memory_bank.half(),
            "projection": projection.cpu(),
            "input_size": 256,
            "feature_grid": 32,
            "seed": args.seed,
        },
        args.model_out,
    )
    print(
        f"Saved {len(memory_bank)} normal patch embeddings from {len(dataset)} images "
        f"to {args.model_out} using {device}."
    )


if __name__ == "__main__":
    main()

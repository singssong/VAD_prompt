#!/usr/bin/env python3
"""Train a one-class patch-feature anomaly detector."""

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
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    def __init__(self, directory):
        self.paths = sorted(
            p for p in Path(directory).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No supported images found in {directory}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
        std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
        return (tensor - mean) / std, path.name


class FeatureExtractor(torch.nn.Module):
    """Wide ResNet feature extractor with deterministic channel subsampling."""

    def __init__(self, channels_per_level=192):
        super().__init__()
        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.channels_per_level = channels_per_level

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        level2 = self.layer2(x)
        level3 = self.layer3(level2)
        level3 = F.interpolate(
            level3, size=level2.shape[-2:], mode="bilinear", align_corners=False
        )
        # Evenly spaced fixed channels retain coverage without a learned projection.
        idx2 = torch.linspace(
            0, level2.shape[1] - 1, self.channels_per_level, device=x.device
        ).long()
        idx3 = torch.linspace(
            0, level3.shape[1] - 1, self.channels_per_level, device=x.device
        ).long()
        features = torch.cat(
            [level2.index_select(1, idx2), level3.index_select(1, idx3)], dim=1
        )
        features = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        return F.normalize(features, p=2, dim=1)


def extract_patches(model, images):
    features = model(images)
    return features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="./data/train")
    parser.add_argument("--model-out", default="./outputs/model.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-size", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(4, (len(dataset) + args.batch_size - 1) // args.batch_size),
        pin_memory=device.type == "cuda",
    )
    model = FeatureExtractor().eval().to(device)
    patches = []
    with torch.inference_mode():
        for images, _ in loader:
            patches.append(extract_patches(model, images.to(device)).cpu())
    patches = torch.cat(patches)

    generator = torch.Generator().manual_seed(args.seed)
    count = min(args.memory_size, len(patches))
    indices = torch.randperm(len(patches), generator=generator)[:count]
    memory_bank = patches[indices].contiguous()
    output = Path(args.model_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": memory_bank,
            "channels_per_level": model.channels_per_level,
            "input_size": 256,
            "backbone": "wide_resnet50_2",
            "seed": args.seed,
        },
        output,
    )
    print(
        f"Saved {count} normal patch features from {len(dataset)} images "
        f"to {output} using {device}."
    )


if __name__ == "__main__":
    main()

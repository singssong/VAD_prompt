#!/usr/bin/env python3
"""Build a normal-patch feature memory bank from training images."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, pil_to_tensor, resize


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.files = sorted(
            p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.files:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            image = image.convert("RGB")
            image = resize(image, [256, 256], interpolation=InterpolationMode.BILINEAR)
            tensor = pil_to_tensor(image).float().div_(255.0)
        return normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )


class FeatureExtractor(torch.nn.Module):
    """ImageNet Wide-ResNet-50-2 features at 1/8 and 1/16 resolution."""

    def __init__(self):
        super().__init__()
        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        x = self.layer1(self.stem(images))
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        low = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        high = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = F.normalize(low, dim=1)
        high = F.normalize(high, dim=1)
        return F.normalize(torch.cat([low, high], dim=1), dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patches-per-image", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().to(device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    sampled_features = []

    with torch.inference_mode():
        for batch_index, images in enumerate(loader, start=1):
            features = extractor(images.to(device, non_blocking=True))
            features = features.permute(0, 2, 3, 1).flatten(1, 2).cpu()
            count = min(args.patches_per_image, features.shape[1])
            for image_features in features:
                indices = torch.randperm(image_features.shape[0], generator=generator)[:count]
                sampled_features.append(image_features[indices])
            print(f"\rExtracted {min(batch_index * args.batch_size, len(dataset))}/{len(dataset)}", end="")

    memory_bank = F.normalize(torch.cat(sampled_features), dim=1).half().contiguous()
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": memory_bank,
            "backbone": "wide_resnet50_2",
            "weights": "IMAGENET1K_V2",
            "input_size": [256, 256],
            "feature_grid": [32, 32],
            "patches_per_image": args.patches_per_image,
            "seed": args.seed,
        },
        args.model_out,
    )
    print(f"\nSaved {len(memory_bank)} normal patch features to {args.model_out}")


if __name__ == "__main__":
    main()

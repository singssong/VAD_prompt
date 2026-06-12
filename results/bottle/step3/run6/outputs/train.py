#!/usr/bin/env python3
"""Train spatial normal-feature statistics for one-class anomaly detection."""

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
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image, [256, 256], interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.to_tensor(image)
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image):
        x = self.stem(image)
        feature1 = self.layer1(x)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        size = feature1.shape[-2:]
        feature2 = F.interpolate(feature2, size=size, mode="bilinear", align_corners=False)
        feature3 = F.interpolate(feature3, size=size, mode="bilinear", align_corners=False)
        return torch.cat((feature1, feature2, feature3), dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().to(device)

    total_backbone_channels = 256 + 512 + 1024
    generator = torch.Generator().manual_seed(args.seed)
    channel_indices = torch.randperm(
        total_backbone_channels, generator=generator
    )[:args.channels].sort().values.to(device)

    feature_sum = None
    feature_square_sum = None
    count = 0
    with torch.inference_mode():
        for images in loader:
            features = extractor(images.to(device, non_blocking=True))
            features = features.index_select(1, channel_indices).double()
            batch_sum = features.sum(dim=0)
            batch_square_sum = features.square().sum(dim=0)
            feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
            feature_square_sum = (
                batch_square_sum
                if feature_square_sum is None
                else feature_square_sum + batch_square_sum
            )
            count += features.shape[0]

    mean = feature_sum / count
    variance = feature_square_sum / count - mean.square()
    # A small variance floor prevents nearly constant normal features from dominating.
    variance = variance.clamp_min(1e-4)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "spatial_diagonal_gaussian",
            "backbone": "wide_resnet50_2_imagenet1k_v2",
            "image_size": 256,
            "channel_indices": channel_indices.cpu(),
            "mean": mean.float().cpu(),
            "variance": variance.float().cpu(),
            "train_image_count": count,
        },
        args.model_out,
    )
    print(f"Trained on {count} images; saved model to {args.model_out}")


if __name__ == "__main__":
    main()

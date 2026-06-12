#!/usr/bin/env python3
"""Score test images using the fitted one-class feature model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms(
            crop_size=256, resize_size=256, antialias=True
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), self.paths[index].name


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        x = self.stem(images)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        size = f1.shape[-2:]
        return torch.cat(
            [
                f1,
                F.interpolate(f2, size=size, mode="bilinear", align_corners=False),
                F.interpolate(f3, size=size, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )


def gaussian_blur(maps, sigma=4.0):
    radius = int(3 * sigma)
    coordinates = torch.arange(-radius, radius + 1, device=maps.device, dtype=maps.dtype)
    kernel = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    maps = F.pad(maps, (radius, radius, 0, 0), mode="reflect")
    maps = F.conv2d(maps, kernel.view(1, 1, 1, -1))
    maps = F.pad(maps, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(maps, kernel.view(1, 1, -1, 1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    channel_indices = checkpoint["channel_indices"].to(device)
    mean = checkpoint["mean"].to(device)
    precision = checkpoint["precision"].to(device)
    map_low = checkpoint["map_low"]
    map_high = checkpoint["map_high"]

    dataset = ImageDataset(args.test_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, (len(dataset) + args.batch_size - 1) // args.batch_size),
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().eval().to(device)
    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    image_scores = {}

    with torch.inference_mode():
        for images, names in loader:
            features = extractor(images.to(device, non_blocking=True))[:, channel_indices]
            features = features.permute(0, 2, 3, 1)
            centered = features - mean
            maps = torch.einsum(
                "bhwd,df,bhwf->bhw", centered, precision, centered
            ).clamp_min_(0).sqrt_()
            maps = F.interpolate(
                maps[:, None], size=(256, 256), mode="bilinear", align_corners=False
            )
            maps = gaussian_blur(maps)

            flat = maps.flatten(1)
            k = max(1, int(flat.shape[1] * 0.01))
            scores = flat.topk(k, dim=1).values.mean(dim=1)
            png_maps = ((maps - map_low) / max(map_high - map_low, 1e-6))
            png_maps = (png_maps.clamp(0, 1) * 255).round().byte().cpu().numpy()

            for name, score, pixel_map in zip(names, scores.cpu().tolist(), png_maps):
                image_scores[name] = float(score)
                Image.fromarray(pixel_map[0], mode="L").save(pixel_dir / name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as file:
        json.dump(image_scores, file, indent=2, sort_keys=True)
        file.write("\n")
    print(f"Scored {len(image_scores)} images on {device}")


if __name__ == "__main__":
    main()

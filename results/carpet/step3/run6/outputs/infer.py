#!/usr/bin/env python3
"""Score test images with a trained one-class patch-memory model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNet18Features(nn.Module):
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


def load_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = TF.resize(
            image, [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True
        )
        tensor = TF.to_tensor(image)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("./data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("./outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    extractor = ResNet18Features().to(device)
    projection = checkpoint["projection"].to(device)
    memory_t = checkpoint["memory_bank"].to(device).T.contiguous()
    low = checkpoint["calibration_low"]
    high = checkpoint["calibration_high"]

    paths = sorted(
        p
        for p in args.test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No images found in {args.test_dir}")

    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}

    with torch.inference_mode():
        for index, path in enumerate(paths, start=1):
            image = load_image(path).to(device)
            features = extractor(image)
            patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
            patches = F.normalize(patches @ projection, dim=1)
            similarity = patches @ memory_t
            distances = torch.sqrt(
                (2.0 - 2.0 * similarity.max(dim=1).values.clamp(-1.0, 1.0)).clamp_min(0.0)
            )

            patch_map = distances.reshape(1, 1, 32, 32)
            pixel_map = F.interpolate(
                patch_map, size=(256, 256), mode="bilinear", align_corners=False
            )[0, 0]
            top_count = max(1, int(pixel_map.numel() * 0.01))
            image_score = float(torch.topk(pixel_map.flatten(), top_count).values.mean().cpu())
            scores[path.name] = image_score

            display_map = ((pixel_map - low) / (high - low)).clamp(0.0, 1.0)
            png = (display_map.mul(255).round().byte().cpu().numpy())
            Image.fromarray(png, mode="L").save(pixel_dir / path.name)
            print(f"[{index:03d}/{len(paths):03d}] {path.name}: {image_score:.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()

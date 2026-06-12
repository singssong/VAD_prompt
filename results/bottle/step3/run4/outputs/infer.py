#!/usr/bin/env python3
"""Score test images and write image-level and pixel-level anomaly scores."""

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
    def __init__(self, root: Path, transform):
        self.paths = sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        names = [path.name for path in self.paths]
        if len(names) != len(set(names)):
            raise RuntimeError("Test images must have unique filenames")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            return self.transform(image), self.paths[index].name


class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        layer1 = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        size = layer1.shape[-2:]
        layer2 = F.interpolate(layer2, size=size, mode="bilinear", align_corners=False)
        layer3 = F.interpolate(layer3, size=size, mode="bilinear", align_corners=False)
        return torch.cat((layer1, layer2, layer3), dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    image_size = int(checkpoint["image_size"])
    transform = ResNet18_Weights.DEFAULT.transforms(
        crop_size=image_size, resize_size=image_size
    )
    dataset = ImageDataset(args.test_dir, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, len(dataset)),
        pin_memory=args.device.startswith("cuda"),
    )

    backbone = resnet18(weights=None)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    extractor = FeatureExtractor(backbone).to(args.device).eval()
    selected = checkpoint["selected_channels"].to(args.device)
    mean = checkpoint["mean"].to(args.device)
    variance = checkpoint["variance"].to(args.device)
    calibration = float(checkpoint["pixel_calibration"])

    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}
    with torch.inference_mode():
        for images, names in loader:
            features = extractor(images.to(args.device, non_blocking=True))
            features = features.index_select(1, selected).float()
            maps = ((features - mean).square() / variance).mean(dim=1, keepdim=True).sqrt()
            maps = F.avg_pool2d(maps, kernel_size=3, stride=1, padding=1)
            maps = F.interpolate(
                maps, size=(image_size, image_size), mode="bilinear", align_corners=False
            ).squeeze(1)

            flat_maps = maps.flatten(1)
            top_count = max(1, int(flat_maps.shape[1] * 0.01))
            image_scores = flat_maps.topk(top_count, dim=1).values.mean(dim=1)
            for name, anomaly_map, image_score in zip(names, maps, image_scores):
                scores[name] = float(image_score.item())
                encoded = (
                    anomaly_map.div(calibration)
                    .clamp(0, 4)
                    .mul(65535.0 / 4.0)
                    .round()
                    .to(torch.uint16)
                    .cpu()
                    .numpy()
                )
                Image.fromarray(encoded).save(pixel_dir / name)

    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images; outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()

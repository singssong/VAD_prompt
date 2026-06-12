#!/usr/bin/env python3
"""Score test images and write image-level scores plus pixel heatmaps."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        return F.avg_pool2d(torch.cat((layer2, layer3), dim=1), kernel_size=3, stride=1, padding=1)


def load_image(path: Path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def nearest_distances(queries, memory, memory_chunk=4096):
    best = torch.full((len(queries),), -1.0, device=queries.device)
    for start in range(0, len(memory), memory_chunk):
        similarity = queries @ memory[start : start + memory_chunk].T
        best = torch.maximum(best, similarity.max(dim=1).values)
    return (1.0 - best).clamp_min_(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("./data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("./outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    args = parser.parse_args()

    paths = sorted(
        p for p in args.test_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No images found in {args.test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    projection = checkpoint["projection"].to(device)
    feature_mean = checkpoint["feature_mean"].to(device)
    feature_scale = checkpoint["feature_scale"].to(device)
    memory = checkpoint["memory"].to(device=device, dtype=torch.float32)
    model = FeatureExtractor().eval().to(device)

    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}
    low = checkpoint["score_low"]
    high = checkpoint["score_high"]

    with torch.inference_mode():
        for index, path in enumerate(paths, 1):
            image = load_image(path).unsqueeze(0).to(device)
            features = model(image)
            height, width = features.shape[-2:]
            patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
            patches = (patches - feature_mean) / feature_scale
            patches = F.normalize(patches @ projection, dim=1)
            distances = nearest_distances(patches, memory)
            patch_map = distances.reshape(1, 1, height, width)

            # Smooth patch scores before upsampling to the required pixel grid.
            patch_map = F.avg_pool2d(patch_map, kernel_size=3, stride=1, padding=1)
            pixel_map = F.interpolate(
                patch_map, size=(256, 256), mode="bilinear", align_corners=False
            )[0, 0]
            flat = pixel_map.flatten()
            top_count = max(1, int(0.01 * flat.numel()))
            image_score = float(torch.topk(flat, top_count).values.mean().cpu())
            scores[path.name] = image_score

            calibrated = ((pixel_map - low) / (high - low)).clamp(0, 1)
            output = (calibrated * 255).round().to(torch.uint8).cpu().numpy()
            Image.fromarray(output, mode="L").save(pixel_dir / path.name)
            print(f"[{index:03d}/{len(paths):03d}] {path.name}: {image_score:.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()

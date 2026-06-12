#!/usr/bin/env python3
"""Score test images and write image-level and pixel-level anomaly outputs."""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import Compose, Normalize, ToTensor


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            return self.transform(image), path.name


def make_backbone(device):
    model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
    model = create_feature_extractor(
        model, return_nodes={"layer2": "layer2", "layer3": "layer3"}
    )
    return model.eval().to(device)


def patch_features(backbone, images, projection):
    features = backbone(images)
    layer2 = F.avg_pool2d(features["layer2"], 3, stride=1, padding=1)
    layer3 = F.avg_pool2d(features["layer3"], 3, stride=1, padding=1)
    layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear",
                           align_corners=False)
    combined = torch.cat((layer2, layer3), dim=1)
    patches = combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1])
    return patches @ projection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    memory = np.ascontiguousarray(checkpoint["memory_bank"].numpy(), dtype=np.float32)
    projection = checkpoint["projection"].to(device)

    index = faiss.IndexFlatL2(memory.shape[1])
    index.add(memory)
    dataset = ImageDataset(args.test_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=device.type == "cuda")
    backbone = make_backbone(device)

    raw_maps = {}
    image_scores = {}
    with torch.inference_mode():
        for batch_index, (images, names) in enumerate(loader, 1):
            batch_size = len(names)
            features = patch_features(
                backbone, images.to(device, non_blocking=True), projection
            )
            distances, _ = index.search(
                np.ascontiguousarray(features.cpu().numpy(), dtype=np.float32), 1
            )
            maps = np.sqrt(np.maximum(distances[:, 0], 0)).reshape(
                batch_size, checkpoint["grid_size"], checkpoint["grid_size"]
            )
            for name, anomaly_map in zip(names, maps):
                raw_maps[name] = anomaly_map
                # A small upper-tail mean is stable for both tiny and broad defects.
                flat = anomaly_map.ravel()
                top_count = max(1, int(round(flat.size * 0.01)))
                image_scores[name] = float(np.partition(flat, -top_count)[-top_count:].mean())
            print(f"\rScoring test images: {batch_index}/{len(loader)}",
                  end="", flush=True)
    print()

    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    pooled = np.concatenate([value.ravel() for value in raw_maps.values()])
    low, high = np.percentile(pooled, (1.0, 99.5))
    scale = max(float(high - low), 1e-8)
    for name, anomaly_map in raw_maps.items():
        normalized = np.clip((anomaly_map - low) / scale, 0, 1)
        image = Image.fromarray((normalized * 255).round().astype(np.uint8), mode="L")
        image = image.resize((256, 256), Image.Resampling.BILINEAR)
        image = image.filter(ImageFilter.GaussianBlur(radius=4.0))
        image.save(pixel_dir / name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(image_scores.items())), handle, indent=2)
        handle.write("\n")
    print(f"Wrote scores and {len(raw_maps)} pixel maps to {args.output_dir}")


if __name__ == "__main__":
    main()

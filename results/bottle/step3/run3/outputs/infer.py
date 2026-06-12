#!/usr/bin/env python3
"""Score test images with positional normal-feature statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from train import IMAGE_EXTENSIONS, IMAGE_SIZE, WideResNetFeatures, anomaly_maps


class TestDataset(Dataset):
    def __init__(self, root: Path):
        self.files = sorted(
            p
            for p in root.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.files:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            tensor = TF.pil_to_tensor(image).float().div_(255.0)
        tensor = TF.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return tensor, path.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("./data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("./outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    dataset = TestDataset(args.test_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    extractor = WideResNetFeatures().to(device)
    channel_indices = state["channel_indices"].to(device)
    mean = state["mean"].to(device)
    variance = state["variance"].to(device)
    map_low = float(state["map_low"])
    map_high = float(state["map_high"])

    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    scores = {}
    with torch.inference_mode():
        for images, names in loader:
            features = extractor(images.to(device))[:, channel_indices]
            maps = anomaly_maps(features, mean, variance)
            flat_maps = maps.flatten(1)
            top_count = max(1, int(flat_maps.shape[1] * 0.01))
            image_scores = flat_maps.topk(top_count, dim=1).values.mean(dim=1)
            rendered = ((maps - map_low) / (map_high - map_low)).clamp(0, 1).mul(255).byte()
            for name, score, pixel_map in zip(names, image_scores, rendered):
                scores[name] = float(score.item())
                array = pixel_map.squeeze(0).cpu().numpy().astype(np.uint8)
                Image.fromarray(array, mode="L").save(pixel_dir / name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(scores.items())), handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Scored {len(scores)} images into {args.output_dir}")


if __name__ == "__main__":
    main()

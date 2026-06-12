#!/usr/bin/env python3
"""Score test images with a fitted pretrained-CNN feature distribution."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import v2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, files: list[Path]) -> None:
        self.files = files
        self.transform = v2.Compose(
            [
                v2.Resize((256, 256), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.files[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), path.name


def gaussian_blur(maps: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    radius = int(3 * sigma)
    coordinates = torch.arange(-radius, radius + 1, device=maps.device, dtype=maps.dtype)
    kernel = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    maps = F.conv2d(maps, kernel.view(1, 1, 1, -1), padding=(0, radius))
    return F.conv2d(maps, kernel.view(1, 1, -1, 1), padding=(radius, 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--scores-out", type=Path, default=Path("outputs/image_scores.json"))
    parser.add_argument("--maps-out", type=Path, default=Path("outputs/pixel_scores"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = image_files(args.test_dir)
    if not files:
        raise RuntimeError(f"No test images found directly under {args.test_dir}")

    device = torch.device(args.device)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    projection = state["projection"].to(device)
    mean = state["mean"].to(device)
    variance = state["variance"].to(device)

    backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
    extractor = create_feature_extractor(
        backbone, return_nodes={"layer2": "layer2", "layer3": "layer3"}
    ).eval().to(device)

    loader = DataLoader(
        ImageDataset(files),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, (len(files) + args.batch_size - 1) // args.batch_size),
        pin_memory=device.type == "cuda",
    )
    args.maps_out.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}

    with torch.inference_mode():
        for images, names in loader:
            outputs = extractor(images.to(device))
            layer2 = F.normalize(outputs["layer2"], dim=1)
            layer3 = F.interpolate(
                F.normalize(outputs["layer3"], dim=1),
                size=layer2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            features = torch.einsum(
                "bchw,cd->bdhw", torch.cat((layer2, layer3), dim=1), projection
            )
            maps = ((features - mean).square() / variance).mean(dim=1).sqrt()
            maps = F.interpolate(
                maps[:, None], size=(256, 256), mode="bilinear", align_corners=False
            )
            maps = gaussian_blur(maps)

            flat_maps = maps.flatten(1)
            top_count = max(1, int(flat_maps.shape[1] * 0.01))
            image_scores = flat_maps.topk(top_count, dim=1).values.mean(dim=1)

            for name, anomaly_map, image_score in zip(names, maps[:, 0], image_scores):
                scores[name] = float(image_score.item())
                # A fixed normal-data scale preserves comparability across output maps.
                encoded = (anomaly_map / (state["map_scale"] * 2.0) * 255.0).clamp(0, 255)
                Image.fromarray(encoded.byte().cpu().numpy(), mode="L").save(
                    args.maps_out / name
                )

    args.scores_out.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_out.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images and wrote results to {args.scores_out.parent}")


if __name__ == "__main__":
    main()

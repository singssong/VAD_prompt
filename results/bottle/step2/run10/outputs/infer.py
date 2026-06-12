#!/usr/bin/env python3
"""Score test images and write image-level and pixel-level anomaly scores."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import v2
from torchvision.transforms.functional import gaussian_blur


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class TestDataset(Dataset):
    def __init__(self, directory: Path):
        self.paths = image_files(directory)
        if not self.paths:
            raise RuntimeError(f"No images found in {directory}")
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
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), path.name


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        self.stem = torch.nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(self.stem(images))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return torch.cat(
            [
                F.adaptive_avg_pool2d(x1, (32, 32)),
                F.interpolate(x2, size=(32, 32), mode="bilinear", align_corners=False),
                F.interpolate(x3, size=(32, 32), mode="bilinear", align_corners=False),
            ],
            dim=1,
        )


def anomaly_maps(
    features: torch.Tensor, mean: torch.Tensor, precision: torch.Tensor
) -> torch.Tensor:
    batch, channels, height, width = features.shape
    vectors = features.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    delta = vectors - mean.unsqueeze(0)
    squared = torch.einsum("bpc,pcd,bpd->bp", delta, precision, delta)
    maps = squared.clamp_min(0).sqrt().reshape(batch, 1, height, width)
    maps = F.interpolate(maps, size=(256, 256), mode="bilinear", align_corners=False)
    return gaussian_blur(maps, kernel_size=[21, 21], sigma=[4.0, 4.0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    mean = checkpoint["mean"].to(device)
    precision = checkpoint["precision"].to(device)
    channel_indices = checkpoint["channel_indices"].to(device)
    training_pixel_scale = checkpoint["pixel_scale"]

    dataset = TestDataset(args.test_dir)
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
    scores: dict[str, float] = {}
    output_maps: list[tuple[str, torch.Tensor]] = []

    with torch.inference_mode():
        for images, names in loader:
            features = extractor(images.to(device, non_blocking=True))
            maps = anomaly_maps(features[:, channel_indices], mean, precision)
            flat = maps.flatten(1)
            top_count = max(1, flat.shape[1] // 100)
            image_scores = flat.topk(top_count, dim=1).values.mean(dim=1)
            for name, score, anomaly_map in zip(names, image_scores, maps):
                scores[name] = float(score)
                output_maps.append((name, anomaly_map[0].cpu()))

    # One common monotonic scale keeps PNG values comparable between images.
    all_pixels = torch.cat([anomaly_map.flatten() for _, anomaly_map in output_maps])
    pixel_scale = max(
        float(training_pixel_scale),
        float(torch.quantile(all_pixels, 0.995).clamp_min(1e-6)),
    )
    for name, anomaly_map in output_maps:
        normalized = (anomaly_map / pixel_scale).clamp(0, 1)
        pixels = (normalized * 255).round().byte().numpy()
        Image.fromarray(pixels, mode="L").save(pixel_dir / name)

    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores)} images and wrote results to {args.output_dir}.")


if __name__ == "__main__":
    main()

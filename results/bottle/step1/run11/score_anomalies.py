#!/usr/bin/env python3
"""Score aligned product images using only normal reference images."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


class ImageDataset(Dataset):
    def __init__(self, files: list[Path], transform):
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            tensor = self.transform(rgb)
            mask_source = torch.frombuffer(
                bytearray(rgb.convert("L").tobytes()), dtype=torch.uint8
            ).reshape(rgb.height, rgb.width).float() / 255.0
        return tensor, mask_source, path.name


def make_loader(files: list[Path], transform, batch_size: int) -> DataLoader:
    return DataLoader(
        ImageDataset(files, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def extract_features(model, loader: DataLoader, device: torch.device):
    all_features, all_luminance, all_names = [], [], []
    for images, luminance, names in loader:
        outputs = model(images.to(device, non_blocking=True))
        layer1 = F.avg_pool2d(outputs["layer1"], kernel_size=2)
        layer2 = outputs["layer2"]
        # Equalize layer influence before concatenating local descriptors.
        layer1 = F.normalize(layer1, dim=1)
        layer2 = F.normalize(layer2, dim=1)
        features = torch.cat((layer1, layer2), dim=1)
        all_features.append(features.cpu())
        luminance = F.interpolate(
            luminance[:, None], size=features.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
        all_luminance.append(luminance)
        all_names.extend(names)
    return torch.cat(all_features), torch.cat(all_luminance), all_names


def spatial_scores(
    features: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    foreground: torch.Tensor,
) -> torch.Tensor:
    standardized = (features - mean) / scale
    patch_scores = standardized.square().mean(dim=1)
    patch_scores = patch_scores[:, foreground]
    top_k = max(4, math.ceil(patch_scores.shape[1] * 0.02))
    strongest = patch_scores.topk(top_k, dim=1).values
    # Combining a tail mean with its maximum rewards both broad and tiny defects.
    return 0.8 * strongest.mean(dim=1) + 0.2 * strongest[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    train_files = image_files(args.train_dir)
    test_files = image_files(args.test_dir)
    if not train_files or not test_files:
        raise RuntimeError("Both training and test directories must contain images")

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    backbone = wide_resnet50_2(weights=weights)
    model = create_feature_extractor(
        backbone, return_nodes={"layer1": "layer1", "layer2": "layer2"}
    ).eval().to(device)

    train_loader = make_loader(train_files, weights.transforms(), args.batch_size)
    test_loader = make_loader(test_files, weights.transforms(), args.batch_size)
    train_features, train_luminance, _ = extract_features(model, train_loader, device)
    test_features, _, test_names = extract_features(model, test_loader, device)

    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False)
    # A variance floor avoids unstable scores from nearly constant channels.
    variance_floor = std.flatten(1).median(dim=1).values[:, None, None] * 0.10
    scale = torch.maximum(std, variance_floor).clamp_min(1e-4)
    foreground = train_luminance.mean(dim=0) < 0.96

    test_raw = spatial_scores(test_features, mean, scale, foreground)

    # Normalize robustly for a useful [0, 1] severity scale while retaining raw scores.
    log_scores = test_raw.clamp_min(1e-12).log()
    low = torch.quantile(log_scores, 0.05)
    high = torch.quantile(log_scores, 0.95)
    anomaly_scores = ((log_scores - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["filename", "anomaly_score", "raw_score"])
        for name, score, raw in zip(test_names, anomaly_scores, test_raw):
            writer.writerow([name, f"{score.item():.8f}", f"{raw.item():.8f}"])

    metadata = {
        "method": "spatial diagonal Gaussian patch anomaly scoring",
        "backbone": "ImageNet Wide ResNet-50-2",
        "feature_layers": ["layer1", "layer2"],
        "training_images": len(train_files),
        "test_images": len(test_files),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

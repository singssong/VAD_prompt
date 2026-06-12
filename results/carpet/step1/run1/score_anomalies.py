#!/usr/bin/env python3
"""Score test images with a compact PatchCore-style nearest-neighbor model."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), self.paths[index].name


class PatchEncoder(torch.nn.Module):
    """Wide ResNet feature extractor with aligned multi-scale patch features."""

    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool, model.layer1
        )
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)

        # Local averaging makes scores less sensitive to one-pixel weave shifts.
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        features = torch.cat((layer2, layer3), dim=1)

        # PatchCore's channel pooling keeps the memory bank compact.
        batch, channels, height, width = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, channels)
        features = F.adaptive_avg_pool1d(features.unsqueeze(1), 384).squeeze(1)
        features = F.normalize(features, dim=1)
        return features.reshape(batch, height * width, 384)


def image_paths(directory: Path) -> list[Path]:
    valid = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in valid)


@torch.inference_mode()
def build_memory_bank(
    encoder: PatchEncoder,
    loader: DataLoader,
    device: torch.device,
    bank_size: int,
) -> torch.Tensor:
    batches = []
    for images, _ in loader:
        batches.append(encoder(images.to(device)).flatten(0, 1).cpu())
    patches = torch.cat(batches)

    generator = torch.Generator().manual_seed(0)
    if len(patches) > bank_size:
        indices = torch.randperm(len(patches), generator=generator)[:bank_size]
        patches = patches[indices]
    return patches.to(device)


@torch.inference_mode()
def score_images(
    encoder: PatchEncoder,
    loader: DataLoader,
    memory_bank: torch.Tensor,
    device: torch.device,
) -> list[tuple[str, float]]:
    results = []
    memory_t = memory_bank.T.contiguous()
    for images, names in loader:
        patches = encoder(images.to(device))
        for image_patches, name in zip(patches, names):
            # Unit-normalized vectors: squared Euclidean NN distance is 2-2*cosine.
            max_similarity = torch.full(
                (len(image_patches),), -1.0, device=device
            )
            for start in range(0, len(memory_bank), 8192):
                similarities = image_patches @ memory_t[:, start : start + 8192]
                max_similarity = torch.maximum(
                    max_similarity, similarities.max(dim=1).values
                )
            distances = (2.0 - 2.0 * max_similarity).clamp_min(0).sqrt()

            # Average the strongest 1% of patch responses to suppress isolated noise.
            k = max(1, int(round(0.01 * len(distances))))
            score = distances.topk(k).values.mean().item()
            results.append((name, score))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--bank-size", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if not args.train_dir.is_dir() or not args.test_dir.is_dir():
        raise FileNotFoundError("Training or test image directory does not exist")

    train_paths = image_paths(args.train_dir)
    test_paths = image_paths(args.test_dir)
    if not train_paths or not test_paths:
        raise RuntimeError("Training and test directories must both contain images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    transform = weights.transforms(crop_size=256, resize_size=256)
    train_loader = DataLoader(
        ImageDataset(train_paths, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ImageDataset(test_paths, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    encoder = PatchEncoder().eval().to(device)
    memory_bank = build_memory_bank(
        encoder, train_loader, device, bank_size=args.bank_size
    )
    scores = score_images(encoder, test_loader, memory_bank, device)

    with args.output.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("filename", "anomaly_score"))
        writer.writerows((name, f"{score:.8f}") for name, score in scores)

    values = np.array([score for _, score in scores])
    print(f"Wrote {len(scores)} scores to {args.output}")
    print(
        f"Score range: {values.min():.6f} to {values.max():.6f}; "
        f"median: {np.median(values):.6f}"
    )
    print("Method: PatchCore-style patch nearest-neighbor anomaly detection")
    print("Backbone: ImageNet-pretrained Wide ResNet-50-2")


if __name__ == "__main__":
    main()

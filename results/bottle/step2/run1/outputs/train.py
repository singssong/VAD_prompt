#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, files):
        self.files = files
        self.transform = Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms(
            crop_size=256, resize_size=256
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB"))


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        x = self.stem(images)
        low = self.layer1(x)
        high = self.layer2(low)
        low = F.adaptive_avg_pool2d(low, high.shape[-2:])
        features = torch.cat((low, high), dim=1)
        return F.normalize(features, dim=1)


def list_images(directory):
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


@torch.inference_mode()
def extract_features(model, files, batch_size, device):
    loader = DataLoader(
        ImageDataset(files), batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda"
    )
    batches = []
    for images in loader:
        batches.append(model(images.to(device, non_blocking=True)).cpu())
    return torch.cat(batches)


def project(features, projection):
    # B,C,H,W -> B,H,W,D
    patches = features.permute(0, 2, 3, 1)
    patches = patches @ projection
    return F.normalize(patches, dim=-1)


def nearest_distances(queries, memory, device, query_chunk=512):
    memory = memory.to(device)
    output = []
    for start in range(0, len(queries), query_chunk):
        query = queries[start:start + query_chunk].to(device)
        # Unit vectors: squared Euclidean distance = 2 - 2*cosine similarity.
        similarity = query @ memory.T
        distance = torch.sqrt(torch.clamp(2.0 - 2.0 * similarity.max(dim=1).values, min=0))
        output.append(distance.cpu())
    return torch.cat(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-size", type=int, default=50000)
    parser.add_argument("--projection-dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = list_images(args.train_dir)
    if len(files) < 5:
        raise RuntimeError(f"Expected at least 5 training images in {args.train_dir}")

    shuffled = files.copy()
    random.shuffle(shuffled)
    calibration_count = max(10, round(0.1 * len(shuffled)))
    calibration_files = shuffled[:calibration_count]
    memory_files = shuffled[calibration_count:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor().eval().to(device)
    memory_features = extract_features(
        model, memory_files, args.batch_size, device
    )
    calibration_features = extract_features(
        model, calibration_files, args.batch_size, device
    )

    channels = memory_features.shape[1]
    generator = torch.Generator().manual_seed(args.seed)
    projection = torch.randn(
        channels, args.projection_dim, generator=generator
    ) / np.sqrt(args.projection_dim)

    memory_patches = project(memory_features, projection).reshape(
        -1, args.projection_dim
    )
    if len(memory_patches) > args.memory_size:
        indices = torch.randperm(len(memory_patches), generator=generator)[
            :args.memory_size
        ]
        memory_patches = memory_patches[indices]

    calibration = project(calibration_features, projection)
    height, width = calibration.shape[1:3]
    calibration_distances = nearest_distances(
        calibration.reshape(-1, args.projection_dim),
        memory_patches, device
    ).reshape(len(calibration_files), height, width)
    calibration_image_scores = torch.topk(
        calibration_distances.flatten(1),
        k=max(1, round(0.01 * height * width)),
        dim=1
    ).values.mean(dim=1)

    image_median = calibration_image_scores.median()
    image_mad = (calibration_image_scores - image_median).abs().median()
    pixel_low = torch.quantile(calibration_distances, 0.90)
    pixel_high = torch.quantile(calibration_distances, 0.999)
    if pixel_high <= pixel_low:
        pixel_high = pixel_low + 1e-6

    artifact = {
        "method": "PatchCore-style nearest-neighbor patch memory",
        "backbone": "ImageNet Wide ResNet-50-2",
        "projection": projection,
        "memory_bank": memory_patches,
        "feature_shape": (height, width),
        "image_median": float(image_median),
        "image_scale": float(max(1.4826 * image_mad, torch.tensor(1e-6))),
        "pixel_low": float(pixel_low),
        "pixel_high": float(pixel_high),
        "seed": args.seed,
    }
    torch.save(artifact, args.output_dir / "model.pt")
    print(
        f"Saved {len(memory_patches)} normal patches from {len(memory_files)} images; "
        f"calibrated on {len(calibration_files)} images."
    )


if __name__ == "__main__":
    main()

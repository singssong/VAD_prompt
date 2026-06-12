#!/usr/bin/env python3
"""Build a normal-patch feature memory bank from training images."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, pil_to_tensor, resize

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = resize(image, [256, 256], InterpolationMode.BILINEAR, antialias=True)
            tensor = pil_to_tensor(image).float().div_(255.0)
        tensor = normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return tensor, path.name


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

    def forward(self, images):
        x = self.layer1(self.stem(images))
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer2 = F.avg_pool2d(layer2, 3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, 3, stride=1, padding=1)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([layer2, layer3], dim=1)


def image_paths(directory):
    return sorted(
        path for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_projection(input_dim, output_dim, seed):
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    projection /= torch.linalg.vector_norm(projection, dim=0, keepdim=True).clamp_min(1e-12)
    return projection


def project_patches(feature_map, projection):
    patches = feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])
    patches = patches @ projection
    return F.normalize(patches, dim=1)


def nearest_distances(queries, bank, query_chunk=4096, bank_chunk=8192):
    output = []
    for start in range(0, len(queries), query_chunk):
        query = queries[start:start + query_chunk]
        best = torch.full((len(query),), -1.0, device=query.device)
        for bank_start in range(0, len(bank), bank_chunk):
            similarities = query @ bank[bank_start:bank_start + bank_chunk].T
            best = torch.maximum(best, similarities.max(dim=1).values)
        output.append(1.0 - best)
    return torch.cat(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="./data/train")
    parser.add_argument("--model-out", default="./outputs/model.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bank-size", type=int, default=30000)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    paths = image_paths(args.train_dir)
    if not paths:
        raise RuntimeError(f"No training images found in {args.train_dir}")

    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(paths), generator=generator).tolist()
    calibration_count = max(1, round(0.1 * len(paths)))
    calibration_paths = [paths[i] for i in order[:calibration_count]]
    bank_paths = [paths[i] for i in order[calibration_count:]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor().eval().to(device)
    projection = make_projection(1536, args.projection_dim, args.seed).to(device)
    loader = DataLoader(
        ImageDataset(bank_paths), batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda"
    )

    sampled_chunks = []
    candidates_per_batch = max(1, args.bank_size // max(1, len(loader)))
    with torch.inference_mode():
        for images, _ in loader:
            features = extractor(images.to(device, non_blocking=True))
            patches = project_patches(features, projection)
            count = min(candidates_per_batch, len(patches))
            indices = torch.randperm(len(patches), device=device)[:count]
            sampled_chunks.append(patches[indices].cpu())
    bank = torch.cat(sampled_chunks)
    if len(bank) > args.bank_size:
        bank = bank[torch.randperm(len(bank), generator=generator)[:args.bank_size]]
    bank_device = bank.to(device)

    calibration_loader = DataLoader(
        ImageDataset(calibration_paths), batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda"
    )
    calibration_distances = []
    with torch.inference_mode():
        for images, _ in calibration_loader:
            features = extractor(images.to(device, non_blocking=True))
            queries = project_patches(features, projection)
            calibration_distances.append(nearest_distances(queries, bank_device).cpu())
    calibration = torch.cat(calibration_distances)
    low = float(torch.quantile(calibration, 0.50))
    high = float(torch.quantile(calibration, 0.999))
    if high <= low:
        high = low + 1e-6

    output = Path(args.model_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "PatchCore-style nearest-neighbor patch features",
            "backbone": "wide_resnet50_2 IMAGENET1K_V2",
            "feature_layers": ["layer2", "layer3"],
            "image_size": 256,
            "grid_size": 32,
            "projection": projection.cpu(),
            "memory_bank": bank,
            "map_low": low,
            "map_high": high,
            "seed": args.seed,
        },
        output,
    )
    print(f"Saved {len(bank)} normal patch features to {output}")
    print(f"Normal calibration range: {low:.6f} to {high:.6f}")


if __name__ == "__main__":
    main()

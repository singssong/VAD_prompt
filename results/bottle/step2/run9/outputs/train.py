#!/usr/bin/env python3
"""Train a positional PatchCore-style one-class anomaly detector."""

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
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
        std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
        return (tensor - mean) / std


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        model = self.backbone
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        layer2 = model.layer2(x)
        layer3 = model.layer3(layer2)
        layer2 = F.normalize(layer2, dim=1)
        layer3 = F.interpolate(
            F.normalize(layer3, dim=1),
            size=layer2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return torch.cat((layer2, layer3), dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("./data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("./outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, max(0, len(dataset) // 32)),
        pin_memory=args.device.startswith("cuda"),
    )
    extractor = FeatureExtractor().to(args.device)

    all_features = []
    with torch.inference_mode():
        for images in loader:
            features = extractor(images.to(args.device, non_blocking=True))
            all_features.append(features.cpu())

    features = torch.cat(all_features, dim=0)
    full_dim = features.shape[1]
    if args.feature_dim > full_dim:
        raise ValueError(f"feature-dim must be <= {full_dim}")
    generator = torch.Generator().manual_seed(args.seed)
    channel_indices = torch.randperm(full_dim, generator=generator)[:args.feature_dim]
    features = features[:, channel_indices]

    # [normal image, channel, y, x] -> [spatial location, normal image, channel]
    memory = features.permute(2, 3, 0, 1).reshape(-1, len(dataset), args.feature_dim)
    checkpoint = {
        "method": "Positional PatchCore",
        "backbone": "wide_resnet50_2",
        "image_size": 256,
        "feature_grid": list(features.shape[-2:]),
        "channel_indices": channel_indices,
        "memory": memory.contiguous().half(),
        "train_count": len(dataset),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model_out)
    print(
        f"Saved {args.model_out}: {len(dataset)} normal images, "
        f"{memory.shape[0]} locations, {args.feature_dim} channels"
    )


if __name__ == "__main__":
    main()

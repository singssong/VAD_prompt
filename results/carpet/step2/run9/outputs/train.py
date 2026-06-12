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
GRID_SIZE = 32
EMBEDDING_DIM = 384
MEMORY_SIZE = 40000


class ImageDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(
            p for p in Path(root).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.files:
            raise RuntimeError(f"No images found in {root}")
        self.transform = Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms(
            crop_size=256, resize_size=256, antialias=True
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            image = image.convert("RGB")
            return self.transform(image)


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        network = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(
            network.conv1, network.bn1, network.relu, network.maxpool
        )
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        x = self.stem(images)
        feature1 = self.layer1(x)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        features = []
        for feature in (feature1, feature2, feature3):
            feature = F.adaptive_avg_pool2d(feature, (GRID_SIZE, GRID_SIZE))
            # Local averaging makes the representation less sensitive to weave phase.
            feature = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
            features.append(feature)
        return torch.cat(features, dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="./data/train")
    parser.add_argument("--model-out", default="./outputs/model.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(4, max(0, (len(dataset) // 32))), pin_memory=device.type == "cuda"
    )
    extractor = FeatureExtractor().to(device)

    total_channels = 256 + 512 + 1024
    generator = torch.Generator().manual_seed(args.seed)
    channel_indices = torch.randperm(total_channels, generator=generator)[:EMBEDDING_DIM]
    channel_indices_device = channel_indices.to(device)

    feature_batches = []
    with torch.inference_mode():
        for images in loader:
            features = extractor(images.to(device, non_blocking=True))
            features = features.index_select(1, channel_indices_device)
            patches = features.permute(0, 2, 3, 1).reshape(-1, EMBEDDING_DIM)
            patches = F.normalize(patches, dim=1)
            feature_batches.append(patches.cpu())
    train_features = torch.cat(feature_batches, dim=0).float()

    memory_generator = torch.Generator().manual_seed(args.seed + 1)
    selected = torch.randperm(len(train_features), generator=memory_generator)
    selected = selected[:min(MEMORY_SIZE, len(train_features))]
    memory_bank = train_features[selected].half().contiguous()

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "PatchCore random coreset",
            "backbone": "wide_resnet50_2",
            "weights": "IMAGENET1K_V2",
            "grid_size": GRID_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "channel_indices": channel_indices,
            "memory_bank": memory_bank,
            "seed": args.seed,
            "train_count": len(dataset),
        },
        model_out,
    )
    print(
        f"Saved {model_out} from {len(dataset)} normal images "
        f"({len(memory_bank)} memory patches) on {device}."
    )


if __name__ == "__main__":
    main()

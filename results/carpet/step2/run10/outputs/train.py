#!/usr/bin/env python3
"""Build a PatchCore-style normal patch memory bank."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
WEIGHTS = Wide_ResNet50_2_Weights.IMAGENET1K_V2


class FeatureExtractor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = wide_resnet50_2(weights=WEIGHTS)
        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.layer1(self.stem(images))
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((layer2, layer3), dim=1)


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((tensor - mean) / std).unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--patches-per-image", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = image_paths(args.train_dir)
    if not paths:
        raise RuntimeError(f"No training images found in {args.train_dir}")

    extractor = FeatureExtractor().to(device)
    banks = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode():
        for index, path in enumerate(paths, 1):
            features = extractor(load_image(path, device))
            patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
            count = min(args.patches_per_image, patches.shape[0])
            selected = torch.randperm(patches.shape[0], generator=generator, device=device)[:count]
            banks.append(patches[selected].cpu())
            if index % 25 == 0 or index == len(paths):
                print(f"Extracted normal patches: {index}/{len(paths)}", flush=True)

    bank = torch.cat(banks)
    feature_mean = bank.mean(dim=0)
    feature_std = bank.std(dim=0).clamp_min(1e-4)
    bank = (bank - feature_mean) / feature_std
    bank = F.normalize(bank, dim=1)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "PatchCore-style nearest-neighbor patch memory",
            "backbone": "wide_resnet50_2 (ImageNet-1K V2)",
            "memory_bank": bank.to(torch.float16),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "image_size": 256,
            "train_image_count": len(paths),
            "seed": args.seed,
        },
        args.model_out,
    )
    print(f"Saved {len(bank)} normal patches to {args.model_out}")


if __name__ == "__main__":
    main()

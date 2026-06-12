#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import FeatureExtractor, ImageDataset, flatten_features


def parse_args():
    parser = argparse.ArgumentParser(description="Build a normal patch feature bank.")
    parser.add_argument("--train-dir", default="./data/train")
    parser.add_argument("--output", default="./outputs/model.pt")
    parser.add_argument("--patches-per-image", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sampled_features = []

    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            features = flatten_features(extractor(images))
            for image_features in features:
                count = min(args.patches_per_image, image_features.shape[0])
                indices = torch.randperm(
                    image_features.shape[0], generator=generator, device=device
                )[:count]
                sampled_features.append(image_features[indices].cpu())
            print(f"Processed {min(batch_index * args.batch_size, len(dataset))}/{len(dataset)}")

    memory_bank = torch.cat(sampled_features, dim=0).contiguous()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": memory_bank,
            "backbone": "wide_resnet50_2",
            "weights": "IMAGENET1K_V2",
            "image_size": 256,
            "feature_grid": 32,
            "patches_per_image": args.patches_per_image,
            "train_images": len(dataset),
        },
        output_path,
    )
    print(f"Saved {memory_bank.shape[0]} normal patch features to {output_path}")


if __name__ == "__main__":
    main()

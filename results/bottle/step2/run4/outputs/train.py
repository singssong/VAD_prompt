#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import (
    FeatureExtractor,
    ImageDataset,
    anomaly_map,
    image_score,
    make_foreground_mask,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fit a normal-image spatial feature model.")
    parser.add_argument("--train-dir", default="./data/train")
    parser.add_argument("--model-out", default="./outputs/model.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset(args.train_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().to(device)

    count = 0
    feature_sum = None
    feature_square_sum = None
    rgb_sum = torch.zeros(3, 256, 256, dtype=torch.float64)
    with torch.inference_mode():
        for images, raw, _ in loader:
            features = extractor(images.to(device, non_blocking=True)).double().cpu()
            if feature_sum is None:
                feature_sum = features.sum(dim=0)
                feature_square_sum = features.square().sum(dim=0)
            else:
                feature_sum += features.sum(dim=0)
                feature_square_sum += features.square().sum(dim=0)
            rgb_sum += raw.double().sum(dim=0).permute(2, 0, 1) / 255.0
            count += images.shape[0]

    mean = (feature_sum / count).float()
    variance = (feature_square_sum / count - mean.double().square()).float()
    channel_floor = variance.flatten(1).median(dim=1).values[:, None, None] * 0.05
    variance = torch.maximum(variance, channel_floor).clamp_min(1e-6)
    mask = make_foreground_mask((rgb_sum / count).float())

    # Calibrate dense PNG values only from normal training scores.
    calibration_values = []
    image_scores = []
    with torch.inference_mode():
        for images, _, _ in loader:
            features = extractor(images.to(device, non_blocking=True))
            maps = anomaly_map(
                features,
                mean[None].to(device),
                variance[None].to(device),
            ).cpu()
            maps *= mask
            for score_map in maps:
                calibration_values.append(score_map[mask > 0.5].flatten())
                image_scores.append(image_score(score_map, mask))
    all_values = torch.cat(calibration_values)
    map_scale = torch.quantile(all_values, 0.999).clamp_min(1e-6).item()

    output = Path(args.model_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean,
            "variance": variance,
            "foreground_mask": mask,
            "map_scale": map_scale,
            "training_images": count,
            "method": "PaDiM-inspired spatial diagonal Gaussian",
            "backbone": "ImageNet Wide ResNet-50-2",
        },
        output,
    )
    print(f"Fitted {count} normal images on {device}; model saved to {output}")
    print(f"Normal image score range: {min(image_scores):.4f} .. {max(image_scores):.4f}")


if __name__ == "__main__":
    main()

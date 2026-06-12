#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms.functional import gaussian_blur


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
            return self.transform(image.convert("RGB")), self.files[index].name


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
        return F.normalize(torch.cat((low, high), dim=1), dim=1)


def list_images(directory):
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def nearest_distances(queries, memory, query_chunk=512):
    output = []
    for start in range(0, len(queries), query_chunk):
        query = queries[start:start + query_chunk]
        similarity = query @ memory.T
        distances = torch.sqrt(
            torch.clamp(2.0 - 2.0 * similarity.max(dim=1).values, min=0)
        )
        output.append(distances)
    return torch.cat(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    files = list_images(args.test_dir)
    if not files:
        raise RuntimeError(f"No test images found in {args.test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(args.model, map_location="cpu", weights_only=True)
    projection = artifact["projection"].to(device)
    memory = artifact["memory_bank"].to(device)
    feature_height, feature_width = artifact["feature_shape"]

    pixel_dir = args.output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    model = FeatureExtractor().eval().to(device)
    loader = DataLoader(
        ImageDataset(files), batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda"
    )

    scores = {}
    with torch.inference_mode():
        for images, names in loader:
            features = model(images.to(device, non_blocking=True))
            patches = features.permute(0, 2, 3, 1) @ projection
            patches = F.normalize(patches, dim=-1)
            distances = nearest_distances(
                patches.reshape(-1, patches.shape[-1]), memory
            ).reshape(len(names), feature_height, feature_width)

            raw_scores = torch.topk(
                distances.flatten(1),
                k=max(1, round(0.01 * feature_height * feature_width)),
                dim=1
            ).values.mean(dim=1)
            normalized_scores = (
                raw_scores - artifact["image_median"]
            ) / artifact["image_scale"]

            maps = F.interpolate(
                distances[:, None], size=(256, 256),
                mode="bilinear", align_corners=False
            )
            maps = gaussian_blur(maps, kernel_size=[9, 9], sigma=[2.0, 2.0])
            maps = (maps - artifact["pixel_low"]) / (
                artifact["pixel_high"] - artifact["pixel_low"]
            )
            maps = maps.clamp(0, 1)

            for name, score, anomaly_map in zip(names, normalized_scores, maps):
                scores[name] = float(score.cpu())
                output = (anomaly_map[0].cpu().numpy() * 255.0).round().astype(
                    np.uint8
                )
                Image.fromarray(output, mode="L").save(pixel_dir / name)

    with (args.output_dir / "image_scores.json").open("w", encoding="utf-8") as f:
        json.dump(dict(sorted(scores.items())), f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Scored {len(scores)} test images.")


if __name__ == "__main__":
    main()

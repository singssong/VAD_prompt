#!/usr/bin/env python3
"""Train a normal-only synthetic-defect detector and score all test images."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = 256


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def random_mask(rng: random.Random) -> np.ndarray:
    scale = 2
    canvas = Image.new("L", (IMAGE_SIZE * scale, IMAGE_SIZE * scale), 0)
    draw = ImageDraw.Draw(canvas)
    kind = rng.choice(["blob", "blob", "line", "scratch"])

    if kind == "blob":
        cx, cy = [rng.randint(25, IMAGE_SIZE - 25) * scale for _ in range(2)]
        rx, ry = rng.randint(5, 35) * scale, rng.randint(4, 27) * scale
        points = []
        count = rng.randint(8, 15)
        for i in range(count):
            angle = 2.0 * math.pi * i / count
            radius = rng.uniform(0.65, 1.25)
            points.append(
                (cx + math.cos(angle) * rx * radius, cy + math.sin(angle) * ry * radius)
            )
        draw.polygon(points, fill=255)
    else:
        count = rng.randint(1, 4 if kind == "scratch" else 2)
        width = rng.randint(1, 3 if kind == "scratch" else 9) * scale
        for _ in range(count):
            x1, y1 = rng.randint(12, 244) * scale, rng.randint(12, 244) * scale
            length = rng.randint(10, 70) * scale
            angle = rng.uniform(0, 2 * math.pi)
            x2, y2 = x1 + length * math.cos(angle), y1 + length * math.sin(angle)
            mid = ((x1 + x2) / 2 + rng.randint(-10, 10) * scale,
                   (y1 + y2) / 2 + rng.randint(-10, 10) * scale)
            draw.line([(x1, y1), mid, (x2, y2)], fill=255, width=width, joint="curve")

    canvas = canvas.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    if rng.random() < 0.45:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 3.5)))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def synthesize_defect(
    image: np.ndarray, source: np.ndarray, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.15:
        return image.copy(), np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    mask = random_mask(rng)
    result = image.copy()
    mode = rng.randrange(6)

    if mode == 0:
        color = np.array(
            [rng.uniform(0.0, 0.25), rng.uniform(0.0, 0.22), rng.uniform(0.0, 0.22)],
            dtype=np.float32,
        )
        defect = image * rng.uniform(0.05, 0.55) + color
    elif mode == 1:
        color = np.array(
            rng.choice([(0.65, 0.02, 0.03), (0.03, 0.12, 0.65), (0.6, 0.5, 0.15)]),
            dtype=np.float32,
        )
        defect = image * rng.uniform(0.25, 0.75) + color * rng.uniform(0.25, 0.8)
    elif mode == 2:
        shift_y, shift_x = rng.randint(-60, 60), rng.randint(-60, 60)
        defect = np.roll(source, (shift_y, shift_x), axis=(0, 1))
        defect = np.clip(defect * rng.uniform(0.65, 1.35), 0, 1)
    elif mode == 3:
        pil = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255))
        radius = rng.uniform(2.0, 7.0)
        defect = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0
    elif mode == 4:
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(0.08, 0.3), image.shape)
        defect = np.clip(image + noise, 0, 1)
    else:
        # Locally erase the weave while preserving its broad color distribution.
        mean = image.mean(axis=(0, 1), keepdims=True)
        defect = np.clip(mean + np.random.default_rng(rng.randrange(2**32)).normal(
            0, rng.uniform(0.005, 0.035), image.shape), 0, 1)

    alpha = mask[..., None]
    result = result * (1.0 - alpha) + defect * alpha
    return np.clip(result, 0, 1).astype(np.float32), (mask > 0.08).astype(np.float32)


class SyntheticDefectDataset(Dataset):
    def __init__(self, paths: list[Path], samples_per_epoch: int, seed: int):
        self.images = [load_rgb(path) for path in paths]
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + self.epoch * self.samples_per_epoch + index)
        image = self.images[rng.randrange(len(self.images))].copy()
        source = self.images[rng.randrange(len(self.images))]

        k = rng.randrange(4)
        image = np.rot90(image, k).copy()
        source = np.rot90(source, k).copy()
        if rng.random() < 0.5:
            image, source = image[:, ::-1].copy(), source[:, ::-1].copy()
        image = np.clip(image * rng.uniform(0.88, 1.12) + rng.uniform(-0.035, 0.035), 0, 1)

        image, mask = synthesize_defect(image, source, rng)
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        target = torch.from_numpy(mask[None])
        return tensor, target


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()
        self.enc1 = ConvBlock(3, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.bridge = ConvBlock(base * 8, base * 12)
        self.dec4 = ConvBlock(base * 20, base * 8)
        self.dec3 = ConvBlock(base * 12, base * 4)
        self.dec2 = ConvBlock(base * 6, base * 2)
        self.dec1 = ConvBlock(base * 3, base)
        self.head = nn.Conv2d(base, 1, 1)

    @staticmethod
    def up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        x = self.bridge(F.max_pool2d(e4, 2))
        x = self.dec4(self.up(x, e4))
        x = self.dec3(self.up(x, e3))
        x = self.dec2(self.up(x, e2))
        x = self.dec1(self.up(x, e1))
        return self.head(x)


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum((1, 2, 3))
    dice = 1.0 - ((2 * intersection + 1.0) /
                  (probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3)) + 1.0)).mean()
    return bce + dice


def train_model(
    train_paths: list[Path], device: torch.device, epochs: int, checkpoint: Path
) -> UNet:
    dataset = SyntheticDefectDataset(train_paths, samples_per_epoch=max(800, len(train_paths) * 4), seed=137)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=device.type == "cuda")
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        model.train()
        running = 0.0
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = loss_fn(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()
        print(f"epoch {epoch + 1:02d}/{epochs}: loss={running / len(loader):.5f}", flush=True)

    torch.save({"model": model.state_dict(), "method": "synthetic-defect segmentation", "backbone": "U-Net"}, checkpoint)
    return model


@torch.inference_mode()
def score_images(model: nn.Module, paths: list[Path], device: torch.device) -> list[float]:
    model.eval()
    scores = []
    for path in paths:
        image = torch.from_numpy(load_rgb(path).transpose(2, 0, 1)).unsqueeze(0).to(device)
        # Flip test-time augmentation reduces sensitivity to directional training artifacts.
        maps = []
        for dims in [None, (-1,), (-2,)]:
            augmented = image if dims is None else image.flip(dims)
            prediction = model(augmented)
            if dims is not None:
                prediction = prediction.flip(dims)
            maps.append(prediction)
        anomaly_map = torch.stack(maps).mean(0)
        anomaly_map = F.avg_pool2d(anomaly_map, kernel_size=7, stride=1, padding=3)
        flat = anomaly_map.flatten()
        k = max(16, flat.numel() // 500)
        scores.append(float(flat.topk(k).values.mean().cpu()))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output", type=Path, default=Path("anomaly_scores.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("anomaly_model.pt"))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    seed_everything(137)
    train_paths = sorted(args.train_dir.glob("*.png"))
    test_paths = sorted(args.test_dir.glob("*.png"))
    if not train_paths or not test_paths:
        raise RuntimeError("Both train and test directories must contain PNG images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, train={len(train_paths)}, test={len(test_paths)}")
    if args.score_only:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model = UNet().to(device)
        model.load_state_dict(saved["model"])
    else:
        model = train_model(train_paths, device, args.epochs, args.checkpoint)
    scores = score_images(model, test_paths, device)

    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "anomaly_score"])
        writer.writerows((path.name, f"{score:.10f}") for path, score in zip(test_paths, scores))

    if len(scores) != len(test_paths) or not np.isfinite(scores).all():
        raise RuntimeError("Output validation failed")
    print(f"wrote {len(scores)} scores to {args.output}")
    print("method: synthetic-defect segmentation")
    print("backbone: compact U-Net trained from scratch")


if __name__ == "__main__":
    main()

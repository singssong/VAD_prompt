from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = 256
FEATURE_SIZE = 32


def image_files(directory):
    return sorted(
        p for p in Path(directory).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, directory):
        self.paths = image_files(directory)
        if not self.paths:
            raise RuntimeError(f"No images found in {directory}")
        self.transform = Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms(
            crop_size=IMAGE_SIZE, resize_size=IMAGE_SIZE, antialias=True
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            raw = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.uint8)
            tensor = self.transform(image)
        return tensor, torch.from_numpy(raw.copy()), self.paths[index].name


class FeatureExtractor(torch.nn.Module):
    """Frozen multi-scale Wide ResNet feature extractor."""

    def __init__(self):
        super().__init__()
        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

        # Evenly spaced channels keep the fitted model compact and deterministic.
        self.register_buffer("idx1", torch.linspace(0, 255, 64).long())
        self.register_buffer("idx2", torch.linspace(0, 511, 64).long())
        self.register_buffer("idx3", torch.linspace(0, 1023, 64).long())
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)

        f1 = F.adaptive_avg_pool2d(f1[:, self.idx1], (FEATURE_SIZE, FEATURE_SIZE))
        f2 = f2[:, self.idx2]
        f3 = F.interpolate(
            f3[:, self.idx3],
            size=(FEATURE_SIZE, FEATURE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        features = []
        for feature in (f1, f2, f3):
            feature = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
            features.append(F.normalize(feature, dim=1, eps=1e-6))
        return torch.cat(features, dim=1) / np.sqrt(3.0)


def anomaly_map(features, mean, variance):
    z2 = (features - mean).square() / variance
    score = torch.sqrt(z2.mean(dim=1).clamp_min(0))
    return F.interpolate(
        score[:, None],
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def make_foreground_mask(mean_rgb):
    gray = (
        0.299 * mean_rgb[0]
        + 0.587 * mean_rgb[1]
        + 0.114 * mean_rgb[2]
    )
    mask = (gray < 0.96).float()[None, None]
    mask = F.max_pool2d(mask, kernel_size=15, stride=1, padding=7)
    return mask[0, 0]


def image_score(score_map, mask):
    values = score_map[mask > 0.5]
    k = max(1, int(values.numel() * 0.01))
    return values.topk(k).values.mean()

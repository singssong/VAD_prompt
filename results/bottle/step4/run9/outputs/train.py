"""Fit a spatial Gaussian model to ImageNet feature maps from normal images."""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import config


class ImageDataset(Dataset):
    def __init__(self, directory: Path):
        self.paths = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {directory}")
        self.transform = Compose([
            Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.paths[index].name


class FeatureExtractor(torch.nn.Module):
    """Extract and concatenate aligned features from two ResNet depths."""

    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        x = self.stem(images)
        level1 = self.layer1(x)
        level2 = self.layer2(level1)
        level2 = F.interpolate(
            level2, size=level1.shape[-2:], mode="bilinear", align_corners=False
        )
        level1 = F.normalize(level1, dim=1)
        level2 = F.normalize(level2, dim=1)
        return torch.cat((level1, level2), dim=1)


def extract_features(extractor, images):
    """Feature extraction stage, kept separate from modeling and scoring."""
    with torch.inference_mode():
        return extractor(images)


def fit_normal_model(extractor, loader, device):
    """Estimate a diagonal Gaussian independently at each feature-map location."""
    feature_sum = None
    feature_square_sum = None
    count = 0
    for images, _ in loader:
        features = extract_features(extractor, images.to(device)).double()
        batch_sum = features.sum(dim=0)
        batch_square_sum = features.square().sum(dim=0)
        feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
        feature_square_sum = (
            batch_square_sum
            if feature_square_sum is None
            else feature_square_sum + batch_square_sum
        )
        count += features.shape[0]

    mean = feature_sum / count
    variance = feature_square_sum / count - mean.square()
    variance = variance.clamp_min(config.VARIANCE_EPS)
    return mean.float(), variance.float()


def gaussian_kernel(sigma, device):
    radius = max(1, int(round(3 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)
    return kernel[None, None], radius


def score_features(features, mean, variance):
    """Convert feature deviations into a smoothed per-pixel anomaly map."""
    squared_z = (features - mean.unsqueeze(0)).square() / variance.unsqueeze(0)
    maps = squared_z.mean(dim=1, keepdim=True).sqrt()
    kernel, padding = gaussian_kernel(config.GAUSSIAN_SIGMA, maps.device)
    maps = F.conv2d(maps, kernel, padding=padding)
    return F.interpolate(
        maps,
        size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def aggregate_image_scores(maps):
    flat = maps.flatten(1)
    top_count = max(1, int(flat.shape[1] * config.TOP_FRACTION))
    return flat.topk(top_count, dim=1).values.mean(dim=1)


def calibrate_scores(extractor, loader, mean, variance, device):
    maps_all = []
    image_scores_all = []
    for images, _ in loader:
        features = extract_features(extractor, images.to(device))
        maps = score_features(features, mean, variance)
        maps_all.append(maps.cpu())
        image_scores_all.append(aggregate_image_scores(maps).cpu())

    map_values = torch.cat(maps_all).flatten()
    image_values = torch.cat(image_scores_all)
    return {
        "pixel_low": torch.quantile(map_values, 0.50).item(),
        "pixel_high": torch.quantile(map_values, 0.999).item(),
        "image_low": torch.quantile(image_values, 0.05).item(),
        "image_high": torch.quantile(image_values, 0.99).item(),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset(config.TRAIN_DIR)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    extractor = FeatureExtractor().to(device)
    mean, variance = fit_normal_model(extractor, loader, device)
    calibration = calibrate_scores(extractor, loader, mean, variance, device)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean.cpu(),
            "variance": variance.cpu(),
            "calibration": calibration,
            "backbone": config.BACKBONE,
            "feature_layers": config.FEATURE_LAYERS,
            "image_size": config.IMAGE_SIZE,
            "training_images": len(dataset),
        },
        config.MODEL_PATH,
    )
    print(f"Saved model trained on {len(dataset)} images to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()


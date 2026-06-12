"""Train a one-class, feature-statistics anomaly detector."""

from pathlib import Path
import math

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.feature_extraction import create_feature_extractor


# Configuration
ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
MODEL_PATH = ROOT / "outputs" / "model.pt"
IMAGE_SIZE = 256
LAYERS = {"layer2": "mid", "layer3": "deep"}
BATCH_SIZE = 16
NUM_WORKERS = 4
EPS = 1e-6
TOP_FRACTION = 0.01
GAUSSIAN_SIGMA = 1.0
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = ResNet50_Weights.IMAGENET1K_V2.transforms(
            crop_size=IMAGE_SIZE, resize_size=IMAGE_SIZE
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            return self.transform(image)


def build_feature_extractor(device: torch.device):
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    extractor = create_feature_extractor(backbone, return_nodes=LAYERS)
    return extractor.eval().to(device)


def extract_features(extractor, images):
    """Fuse mid-level features on the layer2 spatial grid."""
    outputs = extractor(images)
    mid = outputs["mid"]
    deep = F.interpolate(
        outputs["deep"], size=mid.shape[-2:], mode="bilinear", align_corners=False
    )
    return torch.cat((mid, deep), dim=1)


def fit_normal_model(extractor, loader, device):
    """Estimate a diagonal Gaussian independently at every feature location."""
    feature_sum = None
    feature_sq_sum = None
    count = 0
    with torch.inference_mode():
        for images in loader:
            features = extract_features(extractor, images.to(device))
            batch_sum = features.sum(dim=0, dtype=torch.float64).cpu()
            batch_sq_sum = features.square().sum(dim=0, dtype=torch.float64).cpu()
            feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
            feature_sq_sum = (
                batch_sq_sum if feature_sq_sum is None
                else feature_sq_sum + batch_sq_sum
            )
            count += features.shape[0]

    mean = (feature_sum / count).float()
    variance = (feature_sq_sum / count - (feature_sum / count).square()).clamp_min(EPS)
    return {"mean": mean, "inv_std": variance.rsqrt().float(), "count": count}


def gaussian_kernel(sigma, device):
    radius = max(1, math.ceil(3 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)[None, None]


def smooth_maps(maps, sigma):
    kernel = gaussian_kernel(sigma, maps.device)
    padding = kernel.shape[-1] // 2
    return F.conv2d(maps, kernel, padding=padding)


def score_feature_maps(features, mean, inv_std):
    """Return one anomaly value per feature-grid location."""
    standardized = (features - mean) * inv_std
    return standardized.square().mean(dim=1, keepdim=True)


def aggregate_image_scores(pixel_maps, top_fraction):
    flat = pixel_maps.flatten(1)
    top_count = max(1, math.ceil(flat.shape[1] * top_fraction))
    return flat.topk(top_count, dim=1).values.mean(dim=1)


def robust_center_scale(values):
    values = torch.as_tensor(values, dtype=torch.float32)
    center = torch.quantile(values, 0.95)
    lower = torch.quantile(values, 0.50)
    scale = (center - lower).clamp_min(EPS)
    return float(center), float(scale)


def calibrate_normal_scores(extractor, loader, model, device):
    """Calibrate map and image scores using only normal training images."""
    mean = model["mean"].to(device)
    inv_std = model["inv_std"].to(device)
    image_scores = []
    pixel_samples = []
    with torch.inference_mode():
        for images in loader:
            features = extract_features(extractor, images.to(device))
            maps = score_feature_maps(features, mean, inv_std)
            maps = smooth_maps(maps, GAUSSIAN_SIGMA)
            image_scores.extend(aggregate_image_scores(maps, TOP_FRACTION).cpu())
            pixel_samples.append(maps.flatten().cpu())

    pixel_values = torch.cat(pixel_samples)
    image_center, image_scale = robust_center_scale(image_scores)
    pixel_center, pixel_scale = robust_center_scale(pixel_values)
    return {
        "image_center": image_center,
        "image_scale": image_scale,
        "pixel_center": pixel_center,
        "pixel_scale": pixel_scale,
    }


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset(TRAIN_DIR)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda"
    )
    extractor = build_feature_extractor(device)
    model = fit_normal_model(extractor, loader, device)
    model["calibration"] = calibrate_normal_scores(
        extractor, loader, model, device
    )
    model["config"] = {
        "backbone": "resnet50",
        "weights": "IMAGENET1K_V2",
        "image_size": IMAGE_SIZE,
        "layers": list(LAYERS),
        "top_fraction": TOP_FRACTION,
        "gaussian_sigma": GAUSSIAN_SIGMA,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, MODEL_PATH)
    print(f"Trained on {len(dataset)} normal images; saved {MODEL_PATH}")


if __name__ == "__main__":
    main()

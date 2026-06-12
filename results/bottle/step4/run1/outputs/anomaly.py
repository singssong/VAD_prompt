from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, pil_to_tensor, resize

from config import Config


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, directory: Path, image_size: int) -> None:
        self.paths = list_images(directory)
        self.image_size = image_size
        if not self.paths:
            raise RuntimeError(f"No supported images found in {directory}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = resize(
                image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = pil_to_tensor(image).float().div_(255.0)
        tensor = normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return tensor, path.name


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, layers: Iterable[str]) -> None:
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.layers = tuple(layers)
        valid_layers = {"layer1", "layer2", "layer3", "layer4"}
        if len(self.layers) < 2 or not set(self.layers).issubset(valid_layers):
            raise ValueError("Select at least two ResNet feature layers")
        self._features: dict[str, Tensor] = {}
        for name in self.layers:
            getattr(self.backbone, name).register_forward_hook(self._capture(name))
        self.backbone.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _capture(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            self._features[name] = output
        return hook

    def forward(self, images: Tensor) -> Tensor:
        self._features.clear()
        self.backbone(images)
        target_size = max(
            (feature.shape[-2:] for feature in self._features.values()),
            key=lambda size: size[0] * size[1],
        )
        aligned = [
            feature if feature.shape[-2:] == target_size else F.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=False
            )
            for name in self.layers
            for feature in [self._features[name]]
        ]
        return torch.cat(aligned, dim=1)


def extract_features(model: nn.Module, images: Tensor) -> Tensor:
    """Extract concatenated mid-level feature maps."""
    with torch.inference_mode():
        return model(images)


def build_normal_model(feature_batches: Iterable[Tensor], cfg: Config) -> dict[str, Tensor]:
    """Fit a per-location diagonal Gaussian to normal feature maps."""
    count = 0
    feature_sum: Tensor | None = None
    feature_sq_sum: Tensor | None = None
    for features in feature_batches:
        features = features.float().cpu()
        batch_count = features.shape[0]
        batch_sum = features.sum(dim=0)
        batch_sq_sum = features.square().sum(dim=0)
        feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
        feature_sq_sum = batch_sq_sum if feature_sq_sum is None else feature_sq_sum + batch_sq_sum
        count += batch_count
    if count < 2 or feature_sum is None or feature_sq_sum is None:
        raise RuntimeError("At least two training images are required")
    mean = feature_sum / count
    variance = (feature_sq_sum / count - mean.square()).clamp_min(cfg.variance_floor)
    return {"mean": mean, "variance": variance, "count": torch.tensor(count)}


def gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    radius = max(1, int(3.0 * sigma + 0.5))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)[None, None]


def smooth_maps(maps: Tensor, sigma: float) -> Tensor:
    kernel = gaussian_kernel(sigma, maps.device, maps.dtype)
    padding = kernel.shape[-1] // 2
    return F.conv2d(maps[:, None], kernel, padding=padding)[:, 0]


def score_features(
    features: Tensor,
    normal_model: dict[str, Tensor],
    cfg: Config,
) -> tuple[Tensor, Tensor]:
    """Return smoothed pixel maps and top-tail image anomaly scores."""
    mean = normal_model["mean"].to(features.device)
    variance = normal_model["variance"].to(features.device)
    squared_z = (features.float() - mean).square() / variance
    maps = torch.sqrt(squared_z.mean(dim=1).clamp_min(0.0))
    maps = smooth_maps(maps, cfg.gaussian_sigma)
    top_k = max(1, round(maps[0].numel() * cfg.score_top_fraction))
    image_scores = maps.flatten(1).topk(top_k, dim=1).values.mean(dim=1)
    return maps, image_scores


def normalize_scores(
    raw_scores: Tensor,
    low: float,
    high: float,
    high_value: float,
) -> Tensor:
    scale = max(high - low, 1.0e-8)
    gain = high_value / (1.0 - high_value)
    positive_distance = ((raw_scores - low) / scale).clamp_min(0.0) * gain
    return positive_distance / (1.0 + positive_distance)


def resize_maps(maps: Tensor, image_size: int) -> Tensor:
    return F.interpolate(
        maps[:, None],
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[:, 0]

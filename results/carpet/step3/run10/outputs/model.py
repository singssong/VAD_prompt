from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, pil_to_tensor, resize


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(directory):
    return sorted(
        path for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, directory):
        self.files = image_files(directory)
        if not self.files:
            raise RuntimeError(f"No images found in {directory}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = resize(image, [256, 256], interpolation=InterpolationMode.BILINEAR)
            tensor = pil_to_tensor(image).float().div_(255.0)
        tensor = normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return tensor, path.name


class FeatureExtractor(nn.Module):
    """ImageNet Wide ResNet features at strides 8 and 16."""

    def __init__(self):
        super().__init__()
        network = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            network.conv1, network.bn1, network.relu, network.maxpool
        )
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.requires_grad_(False)
        self.eval()

    def forward(self, images):
        features = self.stem(images)
        features = self.layer1(features)
        layer2 = self.layer2(features)
        layer3 = self.layer3(layer2)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        layer2 = F.normalize(layer2, dim=1)
        layer3 = F.normalize(layer3, dim=1)
        features = torch.cat([layer2, layer3], dim=1)
        return F.normalize(features, dim=1)


def flatten_features(features):
    return features.permute(0, 2, 3, 1).reshape(features.shape[0], -1, features.shape[1])


def gaussian_blur(maps, kernel_size=21, sigma=4.0):
    coordinates = torch.arange(kernel_size, device=maps.device) - kernel_size // 2
    kernel = torch.exp(-(coordinates.float() ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel_2d = torch.outer(kernel, kernel).view(1, 1, kernel_size, kernel_size)
    return F.conv2d(maps, kernel_2d, padding=kernel_size // 2)

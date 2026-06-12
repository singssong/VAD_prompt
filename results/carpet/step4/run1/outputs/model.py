from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import config


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = [Path(path) for path in paths]
        self.mean = ResNet18_Weights.IMAGENET1K_V1.transforms().mean
        self.std = ResNet18_Weights.IMAGENET1K_V1.transforms().std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image,
                [config.IMAGE_SIZE, config.IMAGE_SIZE],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.pil_to_tensor(image).float().div_(255.0)
        tensor = TF.normalize(tensor, self.mean, self.std)
        return tensor, path.name


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        x = self.stem(images)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        return {"layer2": layer2, "layer3": layer3}


@torch.inference_mode()
def extract_features(extractor, images, projection):
    """Extract and concatenate projected mid-level patch features."""
    levels = extractor(images)
    resized = [
        F.interpolate(
            levels[name],
            size=(config.FEATURE_GRID_SIZE, config.FEATURE_GRID_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        for name in config.FEATURE_LAYERS
    ]
    features = torch.cat(resized, dim=1)
    features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    features = F.normalize(features, dim=1)
    features = features @ projection
    features = F.normalize(features, dim=1)
    return features


def make_projection(device):
    generator = torch.Generator(device="cpu").manual_seed(config.SEED)
    source_dim = 128 + 256
    projection = torch.randn(
        source_dim, config.PROJECTION_DIM, generator=generator
    )
    projection /= config.PROJECTION_DIM**0.5
    return projection.to(device)


def list_images(directory):
    return sorted(
        path
        for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )

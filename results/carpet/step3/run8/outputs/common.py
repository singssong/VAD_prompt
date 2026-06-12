from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import resnet18


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(directory):
    directory = Path(directory)
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageDataset(Dataset):
    def __init__(self, directory):
        self.files = image_files(directory)
        if not self.files:
            raise RuntimeError(f"No supported images found in {directory}")
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std, path.name


class FeatureExtractor(nn.Module):
    """Frozen ResNet-18 stem through layer2, returned as 32x32 patch descriptors."""

    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)

        # Bring both semantic scales to the same patch grid before concatenation.
        layer1 = F.avg_pool2d(layer1, kernel_size=2, stride=2)
        layer1 = F.normalize(layer1, dim=1)
        layer2 = F.normalize(layer2, dim=1)
        features = torch.cat((layer1, layer2), dim=1)
        return F.normalize(features, dim=1)


def flatten_features(feature_map):
    return feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])


def nearest_prototype_distance(features, prototypes):
    # Unit-normalized descriptors make 1 - cosine similarity a stable distance.
    return 1.0 - features @ prototypes.T


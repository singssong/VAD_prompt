import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

import config


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms(
            crop_size=config.IMAGE_SIZE,
            resize_size=config.IMAGE_SIZE,
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB").resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            return self.transform(image), self.paths[index].name


def list_images(directory: Path):
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def build_backbone(device):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def extract_features(model, images):
    """Concatenate aligned mid-level layer1 and layer2 patch descriptors."""
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    layer1 = model.layer1(x)
    layer2 = model.layer2(layer1)
    layer1 = F.avg_pool2d(layer1, kernel_size=2, stride=2)
    features = torch.cat((layer1, layer2), dim=1)
    return F.normalize(features, dim=1)


def make_loader(paths, shuffle=False):
    return DataLoader(
        ImageDataset(paths),
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


@torch.inference_mode()
def build_normal_feature_model(model, paths, device):
    """Store a reproducibly sampled memory bank of normal patch descriptors."""
    generator = torch.Generator().manual_seed(config.SEED)
    sampled = []
    for images, _ in make_loader(paths):
        features = extract_features(model, images.to(device, non_blocking=True))
        features = features.permute(0, 2, 3, 1).reshape(features.shape[0], -1, features.shape[1])
        for image_features in features:
            count = min(config.PATCHES_PER_IMAGE, image_features.shape[0])
            indices = torch.randperm(image_features.shape[0], generator=generator)[:count]
            sampled.append(image_features[indices.to(device)].cpu())
    memory_bank = torch.cat(sampled)
    if len(memory_bank) > config.MEMORY_BANK_SIZE:
        indices = torch.randperm(len(memory_bank), generator=generator)[:config.MEMORY_BANK_SIZE]
        memory_bank = memory_bank[indices]
    return memory_bank.contiguous()


def nearest_patch_distances(features, memory_bank):
    patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    minimums = []
    for chunk in patches.split(config.DISTANCE_CHUNK_SIZE):
        # Unit-normalized descriptors: squared Euclidean distance is 2 - 2*cosine.
        similarities = chunk @ memory_bank.T
        minimums.append((2.0 - 2.0 * similarities.max(dim=1).values).clamp_min_(0).sqrt_())
    return torch.cat(minimums).reshape(features.shape[0], features.shape[2], features.shape[3])


def gaussian_kernel(sigma, device):
    radius = int(3 * sigma)
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def smooth_maps(maps, sigma=config.GAUSSIAN_SIGMA):
    kernel = gaussian_kernel(sigma, maps.device)
    radius = len(kernel) // 2
    maps = maps[:, None]
    maps = F.pad(maps, (radius, radius, 0, 0), mode="reflect")
    maps = F.conv2d(maps, kernel.view(1, 1, 1, -1))
    maps = F.pad(maps, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(maps, kernel.view(1, 1, -1, 1))[:, 0]


def aggregate_image_scores(maps):
    flat = maps.flatten(1)
    count = max(1, round(flat.shape[1] * config.TOP_FRACTION))
    return flat.topk(count, dim=1).values.mean(dim=1)


@torch.inference_mode()
def score_images(model, paths, memory_bank, device):
    """Return raw smoothed patch maps and top-tail image anomaly scores."""
    all_maps = []
    all_scores = []
    memory_bank = memory_bank.to(device)
    for images, _ in make_loader(paths):
        features = extract_features(model, images.to(device, non_blocking=True))
        maps = smooth_maps(nearest_patch_distances(features, memory_bank))
        all_maps.append(maps.cpu())
        all_scores.append(aggregate_image_scores(maps).cpu())
    return torch.cat(all_maps), torch.cat(all_scores)


def percentile(values, quantile):
    return float(torch.quantile(values.float(), quantile).item())


def train(train_dir=config.TRAIN_DIR, model_path=config.MODEL_PATH):
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    paths = list_images(Path(train_dir))
    if not paths:
        raise RuntimeError(f"No training images found in {train_dir}")

    generator = torch.Generator().manual_seed(config.SEED)
    order = torch.randperm(len(paths), generator=generator).tolist()
    memory_count = min(config.MEMORY_IMAGES, max(1, len(paths) - 1))
    memory_paths = [paths[index] for index in order[:memory_count]]
    calibration_paths = [paths[index] for index in order[memory_count:]]
    if not calibration_paths:
        calibration_paths = memory_paths[-1:]
        memory_paths = memory_paths[:-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_backbone(device)
    memory_bank = build_normal_feature_model(model, memory_paths, device)
    calibration_maps, calibration_scores = score_images(
        model, calibration_paths, memory_bank, device
    )

    artifact = {
        "memory_bank": memory_bank.half(),
        "image_low": percentile(calibration_scores, 0.50),
        "image_high": percentile(calibration_scores, 0.995),
        "pixel_low": percentile(calibration_maps, 0.50),
        "pixel_high": percentile(calibration_maps, 0.999),
        "image_size": config.IMAGE_SIZE,
        "feature_layers": config.FEATURE_LAYERS,
        "backbone": "resnet18",
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, model_path)
    print(
        f"Saved {len(memory_bank)} normal patch features and calibration "
        f"statistics to {model_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=config.TRAIN_DIR)
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.train_dir, args.model_path)

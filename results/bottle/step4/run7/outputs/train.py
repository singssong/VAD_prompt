import sys

import torch
from torch.utils.data import DataLoader

from anomaly import (
    ImageDataset,
    ResNetFeatureExtractor,
    aggregate_image_scores,
    extract_features,
    fit_normal_model,
    score_features,
)
from config import CONFIG


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def calibrate(model, loader, mean, variance, device):
    pixel_scores = []
    image_scores = []
    for images, _ in loader:
        features = extract_features(
            model, images.to(device, non_blocking=True), CONFIG["feature_size"]
        )
        maps = score_features(
            features, mean, variance, CONFIG["gaussian_sigma"]
        )
        scores = aggregate_image_scores(maps, CONFIG["top_fraction"])
        pixel_scores.append(maps.cpu().flatten())
        image_scores.append(scores.cpu())

    pixels = torch.cat(pixel_scores)
    images = torch.cat(image_scores)
    pixel_q = torch.tensor(CONFIG["pixel_calibration_quantiles"])
    image_q = torch.tensor(CONFIG["image_calibration_quantiles"])
    return torch.quantile(pixels, pixel_q), torch.quantile(images, image_q)


def main():
    torch.manual_seed(0)
    device = choose_device()
    dataset = ImageDataset(
        CONFIG["train_dir"], CONFIG["image_size"], CONFIG["extensions"]
    )
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    model = ResNetFeatureExtractor(pretrained=True).to(device).eval()
    mean, variance, count = fit_normal_model(model, loader, device, CONFIG)
    pixel_range, image_range = calibrate(
        model, loader, mean.to(device), variance.to(device), device
    )

    artifact = {
        "method": "spatial diagonal Gaussian patch distribution",
        "backbone": "ImageNet-pretrained ResNet-18",
        "feature_layers": CONFIG["feature_layers"],
        "feature_size": CONFIG["feature_size"],
        "normal_image_count": count,
        "backbone_state_dict": model.state_dict(),
        "mean": mean.cpu(),
        "variance": variance.cpu(),
        "pixel_range": pixel_range,
        "image_range": image_range,
    }
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    torch.save(artifact, CONFIG["model_path"])
    print(f"Saved model from {count} normal images to {CONFIG['model_path']}")
    print(f"Method: {artifact['method']}")
    print(f"Backbone: {artifact['backbone']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Training failed: {error}", file=sys.stderr)
        raise

import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from anomaly import (
    ImageDataset,
    ResNetFeatureExtractor,
    aggregate_image_scores,
    extract_features,
    normalize_scores,
    score_features,
)
from config import CONFIG


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(CONFIG["model_path"], map_location="cpu", weights_only=True)
    model = ResNetFeatureExtractor(pretrained=False)
    model.load_state_dict(artifact["backbone_state_dict"])
    model = model.to(device).eval()
    mean = artifact["mean"].to(device)
    variance = artifact["variance"].to(device)

    dataset = ImageDataset(
        CONFIG["test_dir"], CONFIG["image_size"], CONFIG["extensions"]
    )
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    pixel_dir = CONFIG["output_dir"] / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    image_results = {}

    for images, names in loader:
        features = extract_features(
            model, images.to(device, non_blocking=True), CONFIG["feature_size"]
        )
        maps = score_features(
            features, mean, variance, CONFIG["gaussian_sigma"]
        )
        raw_image_scores = aggregate_image_scores(maps, CONFIG["top_fraction"])
        image_scores = normalize_scores(
            raw_image_scores, artifact["image_range"][0], artifact["image_range"][1]
        )
        maps = normalize_scores(
            maps, artifact["pixel_range"][0], artifact["pixel_range"][1]
        )
        maps = F.interpolate(
            maps[:, None],
            size=(CONFIG["image_size"], CONFIG["image_size"]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        for name, score, anomaly_map in zip(names, image_scores, maps):
            image_results[name] = float(score.cpu())
            png = np.rint(anomaly_map.cpu().numpy() * 255.0).astype(np.uint8)
            Image.fromarray(png, mode="L").save(pixel_dir / name)

    score_path = CONFIG["output_dir"] / "image_scores.json"
    with score_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(image_results.items())), handle, indent=2)
        handle.write("\n")
    print(f"Scored {len(image_results)} images")
    print("Method: spatial diagonal Gaussian patch distribution")
    print("Backbone: ImageNet-pretrained ResNet-18 (layer1 + layer2)")


if __name__ == "__main__":
    main()

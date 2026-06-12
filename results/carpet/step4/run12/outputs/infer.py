from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

import config
from anomaly import (
    ImageDataset,
    MidLevelResNet18,
    aggregate_image_score,
    extract_feature_map,
    gaussian_smooth,
    robust_normalize,
    score_feature_map,
    seed_everything,
)


def main() -> None:
    seed_everything(config.SEED)
    config.PIXEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    memory = checkpoint["memory_bank"].to(device)
    backbone = MidLevelResNet18().to(device)
    dataset = ImageDataset(config.TEST_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    names: list[str] = []
    raw_maps: list[np.ndarray] = []
    with torch.inference_mode():
        for images, batch_names in loader:
            features = extract_feature_map(backbone, images.to(device))
            anomaly_map = score_feature_map(features, memory)
            anomaly_map = gaussian_smooth(anomaly_map, config.GAUSSIAN_SIGMA)
            anomaly_map = F.interpolate(
                anomaly_map[:, None],
                size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            names.append(batch_names[0])
            raw_maps.append(anomaly_map[0].cpu().numpy())

    maps = np.stack(raw_maps)
    low_p, high_p = config.NORMALIZATION_PERCENTILES
    normalized_maps = robust_normalize(maps, low_p, high_p)

    raw_image_scores = np.asarray(
        [aggregate_image_score(pixel_map) for pixel_map in maps], dtype=np.float32
    )
    normalized_image_scores = robust_normalize(raw_image_scores, low_p, high_p)
    scores = {
        name: float(score) for name, score in zip(names, normalized_image_scores)
    }

    with config.SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for name, pixel_map in zip(names, normalized_maps):
        output = np.rint(pixel_map * 255.0).astype(np.uint8)
        Image.fromarray(output, mode="L").save(config.PIXEL_OUTPUT_DIR / name)

    print(f"Scored {len(names)} images and wrote outputs to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

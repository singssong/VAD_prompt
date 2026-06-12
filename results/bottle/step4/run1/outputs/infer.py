#!/usr/bin/env python3
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from anomaly import (
    ImageDataset,
    ResNetFeatureExtractor,
    extract_features,
    normalize_scores,
    resize_maps,
    score_features,
)
from config import CFG


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_data = torch.load(CFG.model_path, map_location="cpu", weights_only=True)
    extractor = ResNetFeatureExtractor(tuple(model_data["layers"])).to(device)
    dataset = ImageDataset(CFG.test_dir, CFG.image_size)
    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    CFG.pixel_dir.mkdir(parents=True, exist_ok=True)
    scores_by_name: dict[str, float] = {}

    score_low = float(model_data["score_low"])
    score_high = float(model_data["score_high"])
    map_low = float(model_data["map_low"])
    map_high = float(model_data["map_high"])

    for images, names in loader:
        features = extract_features(extractor, images.to(device, non_blocking=True))
        maps, raw_scores = score_features(features, model_data, CFG)
        image_scores = normalize_scores(
            raw_scores,
            score_low,
            score_high,
            CFG.normalization_high_value,
        ).cpu()
        maps = resize_maps(maps, CFG.image_size)
        maps = normalize_scores(
            maps,
            map_low,
            map_high,
            CFG.normalization_high_value,
        ).mul(255).round().byte().cpu()

        for name, score, anomaly_map in zip(names, image_scores, maps, strict=True):
            scores_by_name[name] = float(score)
            Image.fromarray(np.asarray(anomaly_map), mode="L").save(CFG.pixel_dir / name)

    with CFG.scores_path.open("w", encoding="utf-8") as handle:
        json.dump(scores_by_name, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(scores_by_name)} images; outputs saved under {CFG.output_dir}")


if __name__ == "__main__":
    main()

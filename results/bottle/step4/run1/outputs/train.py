#!/usr/bin/env python3
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from anomaly import (
    ImageDataset,
    ResNetFeatureExtractor,
    build_normal_model,
    extract_features,
    score_features,
)
from config import CFG


def make_loader(dataset: ImageDataset, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=shuffle,
        num_workers=CFG.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset(CFG.train_dir, CFG.image_size)
    extractor = ResNetFeatureExtractor(CFG.layers).to(device)

    def feature_batches():
        for images, _ in make_loader(dataset):
            yield extract_features(extractor, images.to(device, non_blocking=True))

    normal_model = build_normal_model(feature_batches(), CFG)

    raw_scores: list[torch.Tensor] = []
    map_values: list[torch.Tensor] = []
    for images, _ in make_loader(dataset):
        features = extract_features(extractor, images.to(device, non_blocking=True))
        maps, scores = score_features(features, normal_model, CFG)
        raw_scores.append(scores.cpu())
        map_values.append(maps.flatten().cpu())

    scores = torch.cat(raw_scores)
    values = torch.cat(map_values)
    normal_model["score_low"] = torch.quantile(scores, CFG.score_low_quantile)
    normal_model["score_high"] = torch.quantile(scores, CFG.score_high_quantile)
    normal_model["map_low"] = torch.quantile(values, CFG.map_low_quantile)
    normal_model["map_high"] = torch.quantile(values, CFG.map_high_quantile)
    normal_model["image_size"] = torch.tensor(CFG.image_size)
    normal_model["layers"] = list(CFG.layers)

    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(normal_model, CFG.model_path)
    print(
        f"Saved model from {len(dataset)} normal images to {CFG.model_path} "
        f"(device={device}, feature_shape={tuple(normal_model['mean'].shape)})"
    )


if __name__ == "__main__":
    main()

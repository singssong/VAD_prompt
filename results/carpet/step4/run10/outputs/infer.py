#!/usr/bin/env python
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import CONFIG
from pipeline import (
    build_feature_extractor,
    extract_features,
    image_files,
    make_loader,
    normalize_score,
    score_embeddings,
    set_deterministic,
)


def main() -> None:
    config = CONFIG
    set_deterministic(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = image_files(config.test_dir)
    payload = torch.load(config.model_path, map_location="cpu", weights_only=True)
    projection = payload["projection"].to(device)
    memory_bank = payload["memory_bank"].to(device)
    calibration = payload["calibration"]
    extractor = build_feature_extractor(config, device)

    config.pixel_scores_dir.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}
    loader = make_loader(files, config)
    processed = 0
    for images, names in loader:
        images = images.to(device, non_blocking=True)
        embeddings = extract_features(extractor, images, projection, config)
        maps, raw_scores = score_embeddings(embeddings, memory_bank, config)

        normalized_maps = normalize_score(
            maps,
            calibration["pixel_low"],
            calibration["pixel_scale"],
        )
        normalized_maps = F.interpolate(
            normalized_maps,
            size=(config.image_size, config.image_size),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        normalized_scores = normalize_score(
            raw_scores,
            calibration["image_low"],
            calibration["image_scale"],
        ).clamp(0.0, 1.0)

        for name, anomaly_map, score in zip(names, normalized_maps, normalized_scores):
            pixel_array = (
                anomaly_map[0].mul(255.0).round().byte().cpu().numpy()
            )
            Image.fromarray(pixel_array, mode="L").save(
                config.pixel_scores_dir / name,
                format="PNG",
            )
            scores[name] = float(score.item())

        processed += len(names)
        print(f"\rScoring test images: {processed}/{len(files)}", end="", flush=True)
    print()

    ordered_scores = {path.name: scores[path.name] for path in files}
    with config.scores_path.open("w", encoding="utf-8") as handle:
        json.dump(ordered_scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved {len(ordered_scores)} image scores to {config.scores_path}")


if __name__ == "__main__":
    main()

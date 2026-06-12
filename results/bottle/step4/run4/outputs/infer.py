import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from pipeline import (
    FeatureExtractor,
    ImageDataset,
    aggregate_image_scores,
    extract_features,
    list_images,
    normalize_scores,
    postprocess_maps,
    save_pixel_map,
    score_feature_maps,
)


@torch.inference_mode()
def score_images(extractor, loader, model, pixel_dir, device):
    mean = model["mean"].to(device)
    variance = model["variance"].to(device)
    channel_indices = model["channel_indices"]
    records = []

    for images, names in loader:
        features = extract_features(
            extractor, images.to(device, non_blocking=True), channel_indices
        )
        maps = postprocess_maps(score_feature_maps(features, mean, variance))
        raw_scores = aggregate_image_scores(maps).cpu().numpy()
        normalized_scores = normalize_scores(
            raw_scores, model["image_score_low"], model["image_score_high"]
        )
        maps = maps.cpu().numpy()
        for name, score, anomaly_map in zip(names, normalized_scores, maps):
            records.append((name, float(score)))
            save_pixel_map(
                pixel_dir / name,
                anomaly_map,
                model["pixel_score_low"],
                model["pixel_score_high"],
            )
    return dict(sorted(records))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=config.TEST_DIR)
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--scores-path", type=Path, default=config.SCORES_PATH)
    parser.add_argument("--pixel-dir", type=Path, default=config.PIXEL_DIR)
    args = parser.parse_args()

    files = list_images(args.test_dir)
    if not files:
        raise RuntimeError(f"No test images found in {args.test_dir}")
    args.pixel_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_path, map_location="cpu", weights_only=True)
    extractor = FeatureExtractor().to(device)
    loader = DataLoader(
        ImageDataset(files),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    scores = score_images(extractor, loader, model, args.pixel_dir, device)
    args.scores_path.write_text(json.dumps(scores, indent=2) + "\n")
    print(f"Scored {len(scores)} images and wrote {args.scores_path}")


if __name__ == "__main__":
    main()

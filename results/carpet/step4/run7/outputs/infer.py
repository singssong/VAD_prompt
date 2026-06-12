import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from train import (
    aggregate_image_scores,
    build_backbone,
    extract_features,
    list_images,
    make_loader,
    nearest_patch_distances,
    smooth_maps,
)


def normalize(values, low, high):
    scale = max(float(high) - float(low), 1e-8)
    return ((values - float(low)) / scale).clamp(0.0, 1.0)


@torch.inference_mode()
def score(model, paths, artifact, device):
    """Score test images against the stored normal-feature distribution."""
    memory_bank = artifact["memory_bank"].float().to(device)
    raw_maps = {}
    raw_scores = {}
    for images, names in make_loader(paths):
        features = extract_features(model, images.to(device, non_blocking=True))
        maps = smooth_maps(nearest_patch_distances(features, memory_bank))
        scores = aggregate_image_scores(maps)
        for name, anomaly_map, image_score in zip(names, maps.cpu(), scores.cpu()):
            raw_maps[name] = anomaly_map
            raw_scores[name] = image_score
    return raw_maps, raw_scores


def save_outputs(raw_maps, raw_scores, artifact, output_dir):
    output_dir = Path(output_dir)
    pixel_dir = output_dir / "pixel_scores"
    pixel_dir.mkdir(parents=True, exist_ok=True)

    image_scores = {}
    for name in sorted(raw_scores):
        normalized_score = normalize(
            raw_scores[name], artifact["image_low"], artifact["image_high"]
        )
        image_scores[name] = float(normalized_score.item())

        anomaly_map = normalize(
            raw_maps[name], artifact["pixel_low"], artifact["pixel_high"]
        )
        anomaly_map = F.interpolate(
            anomaly_map[None, None],
            size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        pixels = (anomaly_map.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(pixels, mode="L").save(pixel_dir / name)

    with (output_dir / "image_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(image_scores, handle, indent=2, sort_keys=True)
        handle.write("\n")


def infer(test_dir=config.TEST_DIR, model_path=config.MODEL_PATH, output_dir=config.OUTPUT_DIR):
    paths = list_images(Path(test_dir))
    if not paths:
        raise RuntimeError(f"No test images found in {test_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    model = build_backbone(device)
    raw_maps, raw_scores = score(model, paths, artifact, device)
    save_outputs(raw_maps, raw_scores, artifact, output_dir)
    print(f"Scored {len(paths)} images and saved results to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=config.TEST_DIR)
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(args.test_dir, args.model_path, args.output_dir)

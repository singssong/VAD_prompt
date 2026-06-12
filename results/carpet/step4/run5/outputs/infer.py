import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from common import (
    ImageDataset,
    MidLevelResNet18,
    extract_features,
    gaussian_smooth,
    normalize_features,
    robust_unit_scale,
)
from config import (
    BATCH_SIZE,
    GAUSSIAN_SIGMA,
    IMAGE_SIZE,
    IMAGE_TOP_FRACTION,
    MODEL_PATH,
    NN_QUERY_CHUNK,
    NUM_WORKERS,
    PIXEL_DIR,
    SCORES_PATH,
    TEST_DIR,
)


def nearest_neighbor_distances(queries, memory_bank):
    results = []
    for start in range(0, len(queries), NN_QUERY_CHUNK):
        chunk = queries[start : start + NN_QUERY_CHUNK]
        # Unit-normalized Euclidean distance, computed from cosine similarity.
        similarity = chunk @ memory_bank.T
        results.append(torch.sqrt((2.0 - 2.0 * similarity.max(dim=1).values).clamp_min(0)))
    return torch.cat(results)


def score_images(model, loader, artifact, device):
    memory_bank = artifact["memory_bank"].to(device=device, dtype=torch.float32)
    mean = artifact["feature_mean"].to(device)
    std = artifact["feature_std"].to(device)
    all_maps = []
    filenames = []

    for images, names in loader:
        features = extract_features(model, images.to(device))
        features = normalize_features(features, mean, std)
        batch, height, width, channels = features.shape
        distances = nearest_neighbor_distances(
            features.reshape(-1, channels), memory_bank
        )
        maps = distances.reshape(batch, 1, height, width)
        maps = gaussian_smooth(maps, GAUSSIAN_SIGMA)
        maps = F.interpolate(
            maps,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        all_maps.extend(maps[:, 0].cpu().numpy())
        filenames.extend(names)
    return filenames, np.stack(all_maps)


def aggregate_image_scores(maps):
    flattened = maps.reshape(len(maps), -1)
    top_count = max(1, int(round(flattened.shape[1] * IMAGE_TOP_FRACTION)))
    partitioned = np.partition(flattened, flattened.shape[1] - top_count, axis=1)
    return partitioned[:, -top_count:].mean(axis=1)


def save_scores(filenames, raw_maps):
    PIXEL_DIR.mkdir(parents=True, exist_ok=True)
    scaled_maps, pixel_low, pixel_high = robust_unit_scale(raw_maps)
    raw_image_scores = aggregate_image_scores(raw_maps)
    image_scores, _, _ = robust_unit_scale(
        raw_image_scores, low_percentile=0.0, high_percentile=100.0
    )

    for filename, anomaly_map in zip(filenames, scaled_maps):
        output = Image.fromarray(np.rint(anomaly_map * 255).astype(np.uint8), mode="L")
        output.save(PIXEL_DIR / filename, format="PNG")

    scores = {
        filename: float(score)
        for filename, score in zip(filenames, image_scores)
    }
    with SCORES_PATH.open("w", encoding="utf-8") as file:
        json.dump(scores, file, indent=2, sort_keys=True)
        file.write("\n")
    print(
        f"Saved {len(scores)} scores and maps; pixel scale "
        f"[{pixel_low:.6f}, {pixel_high:.6f}]"
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    dataset = ImageDataset(TEST_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    model = MidLevelResNet18().eval().to(device)
    filenames, raw_maps = score_images(model, loader, artifact, device)
    save_scores(filenames, raw_maps)


if __name__ == "__main__":
    main()

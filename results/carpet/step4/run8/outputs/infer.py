import json

import torch

import config
from pipeline import (
    build_feature_extractor,
    extract_features,
    make_loader,
    normalize_scores,
    save_pixel_map,
    score_features,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normal_model = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    extractor = build_feature_extractor(device)
    loader = make_loader(config.TEST_DIR)
    config.PIXEL_DIR.mkdir(parents=True, exist_ok=True)

    output_scores = {}
    with torch.inference_mode():
        for images, filenames in loader:
            features = extract_features(extractor, images, device)
            maps, raw_scores = score_features(features, normal_model)
            scores = normalize_scores(
                raw_scores,
                normal_model["image_offset"],
                normal_model["image_scale"],
            )
            for filename, anomaly_map, score in zip(filenames, maps, scores):
                output_scores[filename] = float(score.item())
                save_pixel_map(
                    anomaly_map, config.PIXEL_DIR / filename, normal_model
                )

    with config.SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(output_scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Scored {len(output_scores)} images")


if __name__ == "__main__":
    main()

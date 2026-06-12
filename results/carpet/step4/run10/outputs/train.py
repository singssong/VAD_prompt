#!/usr/bin/env python
import torch

from config import CONFIG
from pipeline import (
    build_feature_extractor,
    build_normal_feature_model,
    collect_calibration,
    image_files,
    make_loader,
    make_projection,
    save_model,
    set_deterministic,
)


def main() -> None:
    config = CONFIG
    set_deterministic(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = image_files(config.train_dir)
    print(f"Training on {len(files)} normal images using {device}")

    extractor = build_feature_extractor(config, device)
    input_dim = sum(
        {"layer2": 512, "layer3": 1024}[layer]
        for layer in config.feature_layers
    )
    projection = make_projection(input_dim, config.projection_dim, config.seed).to(device)
    memory_bank = build_normal_feature_model(
        extractor,
        make_loader(files, config),
        projection,
        config,
        device,
    ).to(device)

    generator = torch.Generator().manual_seed(config.seed + 1)
    calibration_count = min(config.calibration_images, len(files))
    calibration_indices = torch.randperm(len(files), generator=generator)[:calibration_count]
    calibration_files = [files[index] for index in calibration_indices.tolist()]
    calibration = collect_calibration(
        extractor,
        make_loader(calibration_files, config),
        projection,
        memory_bank,
        config,
        device,
    )
    save_model(config.model_path, memory_bank, projection, calibration, config)
    print(f"Saved model to {config.model_path}")
    print(f"Memory bank shape: {tuple(memory_bank.shape)}")
    print(f"Calibration: {calibration}")


if __name__ == "__main__":
    main()

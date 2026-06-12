from __future__ import annotations

import torch
from torch.utils.data import DataLoader

import config
from anomaly import ImageDataset, MidLevelResNet18, build_normal_memory, seed_everything


def main() -> None:
    seed_everything(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset(config.TRAIN_DIR)
    loader = DataLoader(
        dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    backbone = MidLevelResNet18().to(device)
    memory = build_normal_memory(backbone, loader, device)

    torch.save(
        {
            "memory_bank": memory,
            "backbone": config.BACKBONE,
            "feature_layers": config.FEATURE_LAYERS,
            "image_size": config.IMAGE_SIZE,
        },
        config.MODEL_PATH,
    )
    print(f"Saved {len(memory)} normal patch descriptors to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()

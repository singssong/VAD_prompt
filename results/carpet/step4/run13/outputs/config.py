from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    train_dir: Path = Path("./data/train")
    test_dir: Path = Path("./data/test_images")
    output_dir: Path = Path("./outputs")
    model_path: Path = Path("./outputs/model.pt")
    scores_path: Path = Path("./outputs/image_scores.json")
    pixel_dir: Path = Path("./outputs/pixel_scores")

    image_size: int = 256
    feature_layers: tuple[str, ...] = ("layer2", "layer3")
    embedding_dim: int = 256
    memory_bank_size: int = 12000
    calibration_fraction: float = 0.12
    batch_size: int = 8
    inference_batch_size: int = 4
    neighbor_chunk_size: int = 3000
    gaussian_sigma: float = 1.25
    top_fraction: float = 0.01
    seed: int = 13


CONFIG = Config()

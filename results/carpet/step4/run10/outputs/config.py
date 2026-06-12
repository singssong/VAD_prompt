from dataclasses import dataclass
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = OUTPUT_DIR.parent


@dataclass(frozen=True)
class Config:
    train_dir: Path = ROOT_DIR / "data" / "train"
    test_dir: Path = ROOT_DIR / "data" / "test_images"
    model_path: Path = OUTPUT_DIR / "normal_model.pt"
    scores_path: Path = OUTPUT_DIR / "image_scores.json"
    pixel_scores_dir: Path = OUTPUT_DIR / "pixel_scores"

    image_size: int = 256
    feature_layers: tuple[str, ...] = ("layer2", "layer3")
    feature_grid_size: int = 32
    projection_dim: int = 384
    memory_bank_size: int = 20000
    calibration_images: int = 48
    batch_size: int = 8
    num_workers: int = 4
    gaussian_sigma: float = 1.5
    image_top_fraction: float = 0.01
    seed: int = 1337


CONFIG = Config()

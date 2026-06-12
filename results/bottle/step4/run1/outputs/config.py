from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[1]
    image_size: int = 256
    layers: tuple[str, ...] = ("layer2", "layer3")
    batch_size: int = 16
    num_workers: int = 4
    variance_floor: float = 1.0e-4
    gaussian_sigma: float = 1.5
    score_top_fraction: float = 0.01
    score_low_quantile: float = 0.50
    score_high_quantile: float = 0.995
    map_low_quantile: float = 0.90
    map_high_quantile: float = 0.999
    normalization_high_value: float = 0.95
    seed: int = 42

    @property
    def train_dir(self) -> Path:
        return self.root / "data" / "train"

    @property
    def test_dir(self) -> Path:
        return self.root / "data" / "test_images"

    @property
    def output_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def model_path(self) -> Path:
        return self.output_dir / "model.pt"

    @property
    def pixel_dir(self) -> Path:
        return self.output_dir / "pixel_scores"

    @property
    def scores_path(self) -> Path:
        return self.output_dir / "image_scores.json"


CFG = Config()

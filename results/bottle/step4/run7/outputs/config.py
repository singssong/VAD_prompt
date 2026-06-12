from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "train_dir": ROOT / "data" / "train",
    "test_dir": ROOT / "data" / "test_images",
    "output_dir": ROOT / "outputs",
    "model_path": ROOT / "outputs" / "normal_model.pt",
    "image_size": 256,
    "feature_layers": ("layer1", "layer2"),
    "feature_size": 64,
    "batch_size": 16,
    "variance_floor": 1e-4,
    "gaussian_sigma": 2.0,
    "top_fraction": 0.01,
    "pixel_calibration_quantiles": (0.50, 0.999),
    "image_calibration_quantiles": (0.50, 0.995),
    "extensions": {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"},
}

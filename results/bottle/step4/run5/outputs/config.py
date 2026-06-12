from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_OUTPUT_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "normal_feature_model.pt"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
FEATURE_SIZE = 64
BACKBONE = "resnet18"
FEATURE_LAYERS = ("layer1", "layer2", "layer3")
BATCH_SIZE = 8
NUM_WORKERS = 2
VARIANCE_EPS = 1e-4
GAUSSIAN_SIGMA = 2.0
IMAGE_TOP_FRACTION = 0.01
CALIBRATION_LOW_QUANTILE = 0.95
CALIBRATION_HIGH_QUANTILE = 0.995
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


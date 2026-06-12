from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_OUTPUT_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "normal_model.pt"

IMAGE_SIZE = 256
FEATURE_SIZE = 32
FEATURE_LAYERS = ("layer1", "layer2", "layer3")
BATCH_SIZE = 16
NUM_WORKERS = 4
VARIANCE_EPS = 1e-4
GAUSSIAN_SIGMA = 1.2
TOP_FRACTION = 0.01
IMAGE_SCALE_QUANTILE = 0.95
PIXEL_SCALE_QUANTILE = 0.995
SEED = 42


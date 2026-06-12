from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_OUTPUT_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "normal_feature_model.pt"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 4
BACKBONE = "resnet50"
FEATURE_LAYERS = ("layer2", "layer3")
FEATURE_SIZE = 32
GAUSSIAN_SIGMA = 1.5
GAUSSIAN_KERNEL_SIZE = 9
VARIANCE_FLOOR = 1.0e-4
TOP_FRACTION = 0.01
IMAGE_SCORE_LOW_QUANTILE = 0.50
IMAGE_SCORE_HIGH_QUANTILE = 0.995
PIXEL_SCORE_LOW_QUANTILE = 0.50
PIXEL_SCORE_HIGH_QUANTILE = 0.999
SEED = 17

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

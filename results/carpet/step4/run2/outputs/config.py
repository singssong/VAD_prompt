from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_OUTPUT_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "normal_model.pt"
IMAGE_SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
BACKBONE = "wide_resnet50_2"
FEATURE_LAYERS = ("layer2", "layer3")
BATCH_SIZE = 8
NUM_WORKERS = 2
SEED = 17

# PatchCore-style normal feature bank settings.
PROJECTION_DIM = 192
MAX_MEMORY_PATCHES = 15000
CALIBRATION_FRACTION = 0.10
DISTANCE_QUERY_CHUNK = 256
DISTANCE_BANK_CHUNK = 3000

# Map and score post-processing.
GAUSSIAN_SIGMA = 1.5
IMAGE_TOP_FRACTION = 0.01
IMAGE_CALIBRATION_QUANTILES = (0.05, 0.995)
PIXEL_CALIBRATION_QUANTILES = (0.01, 0.999)

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

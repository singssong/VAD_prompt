from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_OUTPUT_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "normal_model.pt"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
FEATURE_LAYERS = ("layer2", "layer3")
FEATURE_GRID_SIZE = 32
PROJECTION_DIM = 128
MEMORY_BANK_SIZE = 12000
CALIBRATION_FRACTION = 0.1
BATCH_SIZE = 16
NUM_WORKERS = 4
DISTANCE_QUERY_CHUNK = 256
GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA = 1.0
IMAGE_TOP_FRACTION = 0.01
SEED = 42

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

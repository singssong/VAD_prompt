from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_SCORE_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "model.pt"

IMAGE_SIZE = 256
FEATURE_LAYERS = ("layer1", "layer2")
BATCH_SIZE = 16
MEMORY_IMAGES = 224
CALIBRATION_IMAGES = 56
PATCHES_PER_IMAGE = 160
MEMORY_BANK_SIZE = 35_840
DISTANCE_CHUNK_SIZE = 64
GAUSSIAN_SIGMA = 1.5
TOP_FRACTION = 0.01
SEED = 7

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

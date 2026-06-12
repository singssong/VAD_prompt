from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_OUTPUT_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "normal_memory.pt"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
BACKBONE = "resnet18"
FEATURE_LAYERS = ("layer2", "layer3")
TRAIN_BATCH_SIZE = 8
PATCHES_PER_IMAGE = 96
MEMORY_BANK_SIZE = 20_000
DISTANCE_QUERY_CHUNK = 1024
GAUSSIAN_SIGMA = 1.2
IMAGE_TOP_FRACTION = 0.01
NORMALIZATION_PERCENTILES = (1.0, 99.0)
SEED = 12

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

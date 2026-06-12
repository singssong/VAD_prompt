from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_OUTPUT_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "model.pt"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
BACKBONE = "resnet18"
FEATURE_LAYERS = ("layer2", "layer3")
BATCH_SIZE = 8
NUM_WORKERS = 4
MEMORY_BANK_SIZE = 12_000
GAUSSIAN_SIGMA = 1.5
IMAGE_TOP_FRACTION = 0.01
SEED = 42

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

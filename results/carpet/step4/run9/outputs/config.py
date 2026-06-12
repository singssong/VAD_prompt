from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_OUTPUT_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "model.pt"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
BACKBONE = "resnet18"
FEATURE_LAYERS = ("layer1", "layer2", "layer3")
SELECTED_FEATURES = 160
FEATURE_SIZE = 64
BATCH_SIZE = 16
NUM_WORKERS = 4
SEED = 17
GAUSSIAN_SIGMA = 2.0
TOP_FRACTION = 0.01
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

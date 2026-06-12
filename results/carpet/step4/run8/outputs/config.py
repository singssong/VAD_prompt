from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "normal_model.pt"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 4
BACKBONE = "resnet18"
FEATURE_LAYERS = ("layer2", "layer3")
GAUSSIAN_SIGMA = 1.5
EPSILON = 1e-6
IMAGE_TOP_FRACTION = 0.01
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

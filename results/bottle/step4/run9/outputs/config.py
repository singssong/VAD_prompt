from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "normal_feature_model.pt"
PIXEL_OUTPUT_DIR = OUTPUT_DIR / "pixel_scores"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
BACKBONE = "resnet18"
FEATURE_LAYERS = ("layer1", "layer2")
BATCH_SIZE = 16
NUM_WORKERS = 0
VARIANCE_EPS = 1e-4
GAUSSIAN_SIGMA = 1.5
TOP_FRACTION = 0.01
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


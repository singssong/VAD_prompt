from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_OUTPUT_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "normal_model.pt"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
BACKBONE = "resnet18"
FEATURE_LAYERS = ("layer2", "layer3")
FEATURE_GRID_SIZE = 32
MEMORY_BANK_SIZE = 30000
TRAIN_BATCH_SIZE = 16
INFER_BATCH_SIZE = 1
DISTANCE_QUERY_CHUNK = 256
DISTANCE_BANK_CHUNK = 4096
GAUSSIAN_SIGMA = 4.0
IMAGE_SCORE_TOP_FRACTION = 0.01
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

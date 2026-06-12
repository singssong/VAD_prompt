from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = OUTPUT_DIR.parent
TRAIN_DIR = ROOT_DIR / "data" / "train"
TEST_DIR = ROOT_DIR / "data" / "test_images"
MODEL_PATH = OUTPUT_DIR / "normal_model.pt"
PIXEL_DIR = OUTPUT_DIR / "pixel_scores"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
FEATURE_LAYERS = ("layer1", "layer2")
BATCH_SIZE = 16
NUM_WORKERS = 2
MEMORY_BANK_SIZE = 25_000
CALIBRATION_FRACTION = 0.15
GAUSSIAN_SIGMA = 1.2
IMAGE_SCORE_PERCENTILE = 0.99
RANDOM_SEED = 17
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

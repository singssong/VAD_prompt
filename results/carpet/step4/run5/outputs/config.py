from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = OUTPUT_DIR.parent
TRAIN_DIR = ROOT_DIR / "data" / "train"
TEST_DIR = ROOT_DIR / "data" / "test_images"
MODEL_PATH = OUTPUT_DIR / "feature_bank.pt"
PIXEL_DIR = OUTPUT_DIR / "pixel_scores"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
FEATURE_LAYERS = ("layer2", "layer3")
PATCH_GRID_SIZE = 32
PATCHES_PER_TRAIN_IMAGE = 128
RANDOM_SEED = 17
BATCH_SIZE = 8
NUM_WORKERS = 4
NN_QUERY_CHUNK = 2048
GAUSSIAN_SIGMA = 1.5
IMAGE_TOP_FRACTION = 0.01
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

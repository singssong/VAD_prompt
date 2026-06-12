from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_OUTPUT_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "normal_feature_model.pt"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
BACKBONE = "wide_resnet50_2"
FEATURE_LAYERS = ("layer2", "layer3")
BATCH_SIZE = 8
NUM_WORKERS = 2
MEMORY_BANK_SIZE = 20000
PATCH_POOL_KERNEL = 3
GAUSSIAN_KERNEL = 7
GAUSSIAN_SIGMA = 2.0
IMAGE_SCORE_TOP_FRACTION = 0.01
DISTANCE_QUERY_BATCH = 256
RANDOM_SEED = 12
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

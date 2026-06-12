from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = OUTPUT_DIR.parent
TRAIN_DIR = ROOT_DIR / "data" / "train"
TEST_DIR = ROOT_DIR / "data" / "test_images"
PIXEL_SCORE_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "anomaly_model.pt"
IMAGE_SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
BACKBONE = "wide_resnet50_2"
FEATURE_LAYERS = ("layer2", "layer3")
FEATURE_GRID_SIZE = 32
LOCAL_AVG_POOL_KERNEL = 3

SEED = 17
CALIBRATION_FRACTION = 0.10
PATCHES_PER_TRAIN_IMAGE = 160
MAX_MEMORY_PATCHES = 30000
TRAIN_BATCH_SIZE = 8
INFER_BATCH_SIZE = 1
DISTANCE_QUERY_CHUNK = 256

GAUSSIAN_SIGMA = 1.2
IMAGE_TOP_FRACTION = 0.01
NORMALIZATION_QUANTILE = 0.995

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

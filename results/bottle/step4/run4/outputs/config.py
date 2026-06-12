from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = OUTPUT_DIR.parent
TRAIN_DIR = ROOT_DIR / "data" / "train"
TEST_DIR = ROOT_DIR / "data" / "test_images"
MODEL_PATH = OUTPUT_DIR / "normal_model.pt"
PIXEL_DIR = OUTPUT_DIR / "pixel_scores"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
BACKBONE = "wide_resnet50_2"
FEATURE_LAYERS = ("layer1", "layer2")
SELECTED_CHANNELS = 384
RANDOM_SEED = 17
BATCH_SIZE = 8
NUM_WORKERS = 2
GAUSSIAN_SIGMA = 4.0
GAUSSIAN_KERNEL_SIZE = 25
TOP_FRACTION = 0.01
VARIANCE_EPSILON = 1.0e-4
CALIBRATION_HIGH_QUANTILE = 0.995

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

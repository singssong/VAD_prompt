from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "anomaly_model.pt"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
BACKBONE = "wide_resnet50_2"
FEATURE_LAYERS = ("layer2", "layer3")
BATCH_SIZE = 8
PROJECTION_DIM = 128
MEMORY_BANK_SIZE = 5000
CALIBRATION_FRACTION = 0.20
TOP_FRACTION = 0.01
GAUSSIAN_KERNEL_SIZE = 7
GAUSSIAN_SIGMA = 1.5
SEED = 42

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

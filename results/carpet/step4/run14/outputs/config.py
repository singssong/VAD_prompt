from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = OUTPUT_DIR.parent
TRAIN_DIR = ROOT_DIR / "data" / "train"
TEST_DIR = ROOT_DIR / "data" / "test_images"
MODEL_PATH = OUTPUT_DIR / "model.pt"
IMAGE_SCORES_PATH = OUTPUT_DIR / "image_scores.json"
PIXEL_SCORES_DIR = OUTPUT_DIR / "pixel_scores"

IMAGE_SIZE = 256
BACKBONE = "wide_resnet50_2"
FEATURE_LAYERS = ("layer2", "layer3")
BATCH_SIZE = 8
EMBEDDING_DIM = 192
MEMORY_BANK_SIZE = 24000
CALIBRATION_FRACTION = 0.15
TOP_FRACTION = 0.01
GAUSSIAN_SIGMA = 1.5
KNN_CHUNK_SIZE = 4096
SEED = 17

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

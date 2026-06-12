from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT_DIR = ROOT / "outputs"
PIXEL_OUTPUT_DIR = OUTPUT_DIR / "pixel_scores"
MODEL_PATH = OUTPUT_DIR / "normal_model.pt"
SCORES_PATH = OUTPUT_DIR / "image_scores.json"

IMAGE_SIZE = 256
BACKBONE = "wide_resnet50_2"
FEATURE_LAYERS = ("layer2", "layer3")
FEATURE_GRID_SIZE = 32
PROJECTION_DIM = 256
MEMORY_BANK_SIZE = 12000
TRAIN_BATCH_SIZE = 8
INFER_BATCH_SIZE = 4
NN_QUERY_CHUNK = 1024
GAUSSIAN_SIGMA = 1.5
IMAGE_TOP_FRACTION = 0.01
RANDOM_SEED = 17

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

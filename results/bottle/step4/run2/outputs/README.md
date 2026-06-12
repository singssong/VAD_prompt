# Feature-Based Anomaly Detection

This implementation uses an ImageNet-pretrained Wide ResNet-50-2 backbone. It
concatenates mid-level `layer2` and `layer3` feature maps, projects and
normalizes the patch descriptors, and compares each test patch to a sampled
memory bank of normal training patches. The nearest-neighbor distances form the
pixel anomaly map. Maps are Gaussian-smoothed before resizing to 256x256, and
the image score is the mean of the highest-scoring 1% of pixels.

Both pixel and image scores are normalized to `[0, 1]` using quantiles measured
from the normal training set. A bounded monotonic image-score transform
preserves the ranking of scores beyond the normal range instead of clipping
them. Pixel maps are written as single-channel 8-bit PNG files.

## Run

From the repository root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The torchvision pretrained weights are downloaded automatically if they are
not already in the local PyTorch cache. CUDA is used when available; otherwise
the scripts run on CPU.

Outputs:

- `outputs/anomaly_model.pt`: learned normal patch memory and calibration
- `outputs/image_scores.json`: one normalized anomaly score per test filename
- `outputs/pixel_scores/`: one 256x256 grayscale PNG per test image

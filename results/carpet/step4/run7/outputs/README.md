# Feature-Based Anomaly Detection

This solution uses an ImageNet-pretrained ResNet-18 as a frozen feature
extractor. It concatenates aligned features from `layer1` and `layer2`, samples
normal patch descriptors into a memory bank, and scores each test patch by its
nearest-neighbor distance to that bank. Gaussian-smoothed patch maps produce
pixel scores, and the mean of the highest-scoring 1% of pixels produces the
image score.

Scores are calibrated using a held-out subset of normal training images and
clipped to `[0, 1]`. No labels or masks are used.

## Run

From this directory:

```bash
python -m pip install -r requirements.txt
python train.py
python infer.py
```

The first run may download the official torchvision ResNet-18 ImageNet weights.
CUDA is used automatically when available. Paths and model settings are kept in
`config.py`; command-line path overrides are also available via `--help`.

Inference writes:

- `image_scores.json`
- `pixel_scores/<test filename>` as single-channel 256x256 PNG images

# Feature-Based Anomaly Detection

This implementation uses an ImageNet-pretrained Wide ResNet-50-2 as a frozen
feature extractor. It concatenates projected patch embeddings from `layer2` and
`layer3`, then uses nearest-neighbor distance to a memory bank of normal
training patches as the anomaly signal.

Pixel maps are Gaussian-smoothed before being resized to 256x256. Image scores
are the mean of the highest-scoring 1% of patches and are mapped to `[0, 1]`
using calibration statistics from held-out normal training images.

## Run

From the repository root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run may download torchvision's ImageNet weights if they are not
already cached. Training writes `outputs/model.pt`. Inference writes
`outputs/image_scores.json` and one grayscale PNG per test image under
`outputs/pixel_scores/`.

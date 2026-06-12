# Feature-Based Anomaly Detection

This solution uses an ImageNet-pretrained ResNet-18 and concatenates mid-level
features from `layer1`, `layer2`, and `layer3`. A deterministic channel subset is
modeled with a spatial diagonal Gaussian fitted only on normal training images.
Per-location standardized distances form the anomaly map. The map is Gaussian
smoothed, resized to 256x256, and its highest-scoring 1% of pixels are averaged
for the image score. Training statistics calibrate image scores to `[0, 1]`.

Run from this directory:

```bash
python train.py
python infer.py
```

Or run from the repository root:

```bash
python outputs/train.py
python outputs/infer.py
```

The first run may download the standard torchvision ResNet-18 ImageNet weights.
Training writes `model.pt`. Inference writes `image_scores.json` and one
single-channel 256x256 PNG per test image under `pixel_scores/`.

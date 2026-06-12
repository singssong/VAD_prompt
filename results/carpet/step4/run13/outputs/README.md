# Feature-Based Anomaly Detection

This implementation uses an ImageNet-pretrained Wide ResNet-50-2 backbone and
extracts patch features from `layer2` and `layer3`. The aligned features are
concatenated, randomly projected, and compared with a memory bank of normal
training patches using nearest-neighbor distance. Pixel maps are Gaussian
smoothed and image scores use the mean of the highest-scoring 1% of patches.

Run from the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

Training writes `outputs/model.pt`. Inference writes:

- `outputs/image_scores.json`
- one 256x256 single-channel PNG per test image in `outputs/pixel_scores/`

Image and pixel scores are normalized with robust ranges estimated from a
held-out subset of the normal training images. Image scores use a bounded
non-saturating transform so stronger anomalies retain their relative ordering.

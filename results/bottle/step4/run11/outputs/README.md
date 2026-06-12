# Feature-Based Bottle Anomaly Detection

This implementation uses a PatchCore-style detector. ImageNet-pretrained
Wide ResNet-50-2 features from `layer2` and `layer3` are aligned and
concatenated, randomly projected, and compared with a memory bank of normal
training patches by nearest-neighbor distance.

Anomaly maps are Gaussian-smoothed at feature resolution and bilinearly resized
to 256x256. The image score is the mean of the highest 1% of map values and is
mapped to `[0, 1]` by a bounded logistic calibration based on robust quantiles
measured only on normal training data.

## Run

From the repository root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run may download the official torchvision ImageNet weights. Training
creates `outputs/normal_model.pt`. Inference creates `outputs/image_scores.json`
and one single-channel 256x256 PNG per test image under
`outputs/pixel_scores/`.

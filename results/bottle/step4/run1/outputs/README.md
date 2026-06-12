# Feature-Based Anomaly Detection

This pipeline uses an ImageNet-pretrained ResNet-18. It concatenates spatial
features from `layer2` and `layer3`, then fits a per-location diagonal Gaussian
distribution using only normal training images. Anomaly maps are standardized
feature distances with Gaussian smoothing. Image scores are the mean of the
highest-scoring 1% of map locations and are calibrated to `[0, 1]` using normal
training-score quantiles and a bounded monotonic transform that preserves
ranking above the calibration range.

## Run

From the repository root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

Training reads `data/train/` and writes `outputs/model.pt`. Inference reads
`data/test_images/` and writes:

- `outputs/image_scores.json`
- `outputs/pixel_scores/<test filename>`

All inputs are converted to RGB and resized to 256x256 before feature
extraction. Every pixel-score PNG is single-channel and 256x256.

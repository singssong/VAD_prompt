# Feature-based anomaly detection

This implementation uses ImageNet-pretrained ResNet-50 features from `layer2`
and `layer3`. The feature maps are resized to a common grid and concatenated.
Training estimates a diagonal Gaussian at every spatial location from normal
images only. Inference scores standardized feature deviations, smooths the
patch anomaly map with a Gaussian kernel, and aggregates its highest-scoring
1% of locations into an image score.

## Run

From the directory containing `data/` and `outputs/`:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run may download the official torchvision ResNet-50 ImageNet weights.
Training writes `outputs/model.pt`. Inference writes `outputs/image_scores.json`
and one 256x256 grayscale PNG per test image under `outputs/pixel_scores/`.

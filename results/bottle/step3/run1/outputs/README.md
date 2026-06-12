# Feature-Based Anomaly Detection

This solution uses ImageNet-pretrained Wide ResNet-50-2 features from three
stages. Features are resized to a common 32x32 grid, locally averaged, projected
to 128 dimensions, and modeled with an independent Gaussian at each spatial
location. Pixel scores are standardized distances from this normal feature
distribution. The image score is the mean of the highest 1% of smoothed pixel
scores.

All images are resized to 256x256 before feature extraction. No labels or masks
are used.

## Run

From the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run may download the standard torchvision ImageNet weights. Training
creates `outputs/model.pt`. Inference creates `outputs/image_scores.json` and
one 256x256 grayscale PNG per test image in `outputs/pixel_scores/`.

CPU execution is supported with `--device cpu`; CUDA is selected automatically
when available.

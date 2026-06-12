# Bottle Anomaly Detection

This solution uses ImageNet-pretrained Wide ResNet-50-2 features from layers 1,
2, and 3. It learns a diagonal Gaussian distribution of normal features at
each spatial location. Pixel anomaly values are per-location Mahalanobis
distances, and each image score is the mean of the highest 1% of pixel values.

Run from the directory that contains `data/` and `outputs/`:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run may download the standard torchvision ImageNet weights. Training
creates `outputs/model.pt`. Inference creates `outputs/image_scores.json` and
one 256x256 single-channel PNG per test image in `outputs/pixel_scores/`.

CUDA is used automatically when available; CPU execution is also supported.

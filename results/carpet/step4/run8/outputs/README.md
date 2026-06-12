# Feature-based anomaly detection

This implementation uses an ImageNet-pretrained ResNet-18. It concatenates
normalized feature maps from `layer2` and `layer3`, then fits a spatial,
diagonal Gaussian distribution using only the normal training images. Pixel
scores are diagonal Mahalanobis distances, Gaussian-smoothed and resized to
256x256. Image scores are the mean of the highest-scoring 1% of map locations
and are robustly normalized to `[0, 1)` with a monotonic transform calibrated
on normal training scores.

Run from this directory:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The scripts read only `data/train/` and `data/test_images/`. Outputs are written
to `outputs/image_scores.json` and `outputs/pixel_scores/`. The first run may
download torchvision's ImageNet ResNet-18 weights if they are not cached.

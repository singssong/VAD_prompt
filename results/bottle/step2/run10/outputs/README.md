# Bottle Anomaly Detection

This solution uses PaDiM-style spatial Gaussian modeling over fixed ImageNet
features from a Wide ResNet-50-2 backbone. Training uses only normal images.
Each pixel score is a smoothed Mahalanobis distance, and each image score is
the mean of the highest 1% of its pixel scores. Pixel PNGs share a robust
dataset-wide scale, so their grayscale values remain comparable across images.

Run from the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run downloads torchvision's ImageNet pretrained backbone weights.
Training writes `outputs/model.pt`. Inference writes `outputs/image_scores.json`
and one 256x256 grayscale PNG per test image under `outputs/pixel_scores/`.

Optional arguments:

```bash
python outputs/train.py --train-dir data/train --output outputs/model.pt
python outputs/infer.py --test-dir data/test_images --model outputs/model.pt \
  --output-dir outputs
```

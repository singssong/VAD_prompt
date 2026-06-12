# Carpet Anomaly Detection

This solution uses a PaDiM-style one-class feature distribution model. Frozen
ImageNet ResNet-18 features from three scales are sampled, and a regularized
multivariate Gaussian is fitted using only the normal training images. At
inference, patch Mahalanobis distances provide pixel maps and the mean of the
most anomalous 1% of pixels provides each image score.

Run from the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run may download torchvision's ResNet-18 ImageNet weights. Training
creates `outputs/model.pt`. Inference creates `outputs/image_scores.json` and
one 256x256 grayscale PNG per test image in `outputs/pixel_scores/`.

Optional arguments:

```bash
python outputs/train.py --train-dir data/train --output-dir outputs
python outputs/infer.py --test-dir data/test_images --output-dir outputs \
  --model outputs/model.pt
```

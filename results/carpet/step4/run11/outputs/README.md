# Feature-Based Anomaly Detection

This implementation uses a PaDiM-style feature distribution model. An
ImageNet-pretrained Wide ResNet-50-2 extracts `layer2` and `layer3` features.
The maps are spatially aligned and concatenated, then a fixed random subset of
channels is modeled by a regularized multivariate Gaussian at every location.
Mahalanobis distance produces the anomaly map.

The anomaly map is Gaussian-smoothed before resizing to 256x256. The image
score is the mean of the highest 1% of map values. Pixel and image scores are
normalized with a bounded logistic mapping calibrated from robust ranges
measured only from the normal training data.

## Run

From the project root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run may download the official torchvision ImageNet weights. Training
uses only `data/train/`; inference reads only `data/test_images/`.

Outputs:

- `outputs/image_scores.json`
- `outputs/pixel_scores/<test filename>.png`
- `outputs/model.pt`

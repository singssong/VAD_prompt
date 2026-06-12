# Feature-based anomaly detection

This solution uses an ImageNet-pretrained ResNet-18 and concatenates normalized
features from `layer1` and `layer2`. It fits a diagonal Gaussian at every spatial
feature location using only normal training images. At inference, standardized
feature distance produces the anomaly map. The map is Gaussian-smoothed, resized
to 256x256, and its highest-scoring 1% of pixels are averaged for the image score.
Scores are calibrated to `[0, 1]` using the normal training-score distribution.

## Run

From the project directory:

```bash
python outputs/train.py
python outputs/infer.py
```

Install dependencies when needed:

```bash
python -m pip install -r outputs/requirements.txt
```

Training reads only `data/train/`. Inference reads only `data/test_images/`.
The trained statistics are stored in `outputs/normal_feature_model.pt`.
Inference creates `outputs/image_scores.json` and one 256x256 grayscale PNG per
test image in `outputs/pixel_scores/`.

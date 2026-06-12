# Bottle Anomaly Detection

This solution uses frozen ImageNet-pretrained ResNet-18 features from three
intermediate stages. Training estimates a per-spatial-location diagonal
Gaussian distribution of normal features. Inference computes a regularized
diagonal Mahalanobis distance at each feature-grid location, smooths and
resizes the result to 256x256, and uses the mean of the highest-scoring 1% of
pixels as the image anomaly score.

Run from the task root:

```bash
python3 -m pip install -r outputs/requirements.txt
python3 outputs/train.py
python3 outputs/infer.py
```

The first run may download the standard torchvision ResNet-18 ImageNet weights.
Training writes `outputs/model.pt`. Inference writes `outputs/image_scores.json`
and one 8-bit single-channel 256x256 PNG per test image under
`outputs/pixel_scores/`. Higher JSON scores and brighter map pixels indicate
greater anomaly.

Optional arguments:

```bash
python3 outputs/train.py --batch-size 16 --device cuda
python3 outputs/infer.py --batch-size 16 --device cuda
```

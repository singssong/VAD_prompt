# Feature-Based Anomaly Detection

This solution uses ImageNet-pretrained ResNet-18 intermediate patch features.
Features from `layer2` and upsampled `layer3` are fused on a 32x32 grid and
L2-normalized. Training estimates a diagonal Gaussian distribution independently
at each spatial location using only normal images. Inference computes a
per-location Mahalanobis-style deviation, smooths and upsamples it to 256x256,
and uses the mean of the highest-scoring 1% of pixels as the image score.

## Run

Commands are run from the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run may download the ImageNet ResNet-18 weights through torchvision.
The fitted model is saved as `outputs/model.pt`. Inference writes
`outputs/image_scores.json` and 16-bit single-channel PNG maps under
`outputs/pixel_scores/`.

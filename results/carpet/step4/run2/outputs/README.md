# Feature-based anomaly detection

This solution uses a PatchCore-style detector. ImageNet-pretrained
Wide ResNet-50-2 features are taken from `layer2` and `layer3`, locally averaged,
aligned, concatenated, projected, and compared with a sampled bank of normal
training patches using nearest-neighbor distance.

All inputs are resized to 256x256. Patch distances form the anomaly map, which is
Gaussian-smoothed and resized to 256x256. The image score is the mean of the top
1% of pixel scores. Held-out normal training images set fixed robust
normalization ranges; a monotonic sigmoid keeps image scores in `[0, 1]`
without collapsing the ordering of strong anomalies.

Run from this directory or any other working directory:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The scripts use only `data/train/` for model fitting and `data/test_images/` for
inference. Training writes `outputs/normal_model.pt`; inference writes
`outputs/image_scores.json` and the grayscale maps under
`outputs/pixel_scores/`.

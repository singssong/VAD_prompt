# Feature-Based Anomaly Detection

This solution uses an ImageNet-pretrained ResNet-18 as a frozen feature
extractor. Mid-level `layer2` and `layer3` feature maps are spatially aligned
and concatenated. Training estimates normal feature statistics and stores a
sampled normal patch memory bank. Inference assigns each patch its nearest-bank
distance, applies Gaussian smoothing, resizes the map to 256x256, and uses the
mean of the highest-scoring 1% of pixels as the image anomaly score.

Scores are normalized to `[0, 1]`, with larger values indicating stronger
anomalies. Pixel maps are globally robust-normalized and saved as single-channel
8-bit PNG files.

Run from the project directory:

```bash
python outputs/train.py
python outputs/infer.py
```

The scripts read only `data/train/` for model fitting and `data/test_images/`
for inference. The generated model is `outputs/feature_bank.pt`; required
results are written to `outputs/image_scores.json` and
`outputs/pixel_scores/`.

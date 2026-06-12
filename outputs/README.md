# Feature-Based Anomaly Detection

This solution uses an ImageNet-pretrained ResNet-18 as a frozen feature
extractor. Features from `layer2` and `layer3` are spatially aligned,
concatenated, standardized from normal training data, and sampled into a
normal-patch memory bank. Test patches receive the Euclidean distance to their
nearest normal patch. The patch map is Gaussian-smoothed and resized to
256x256. The image score is the mean of the highest-scoring 1% of pixels and is
robustly normalized to `[0, 1]`.

## Run

From the repository root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run may download the official torchvision ResNet-18 ImageNet weights.
Inference writes `outputs/image_scores.json` and one single-channel 256x256 PNG
per test image under `outputs/pixel_scores/`.

All paths and model settings are defined in `outputs/config.py`.

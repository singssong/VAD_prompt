# Feature-based anomaly detection

This implementation uses an ImageNet-pretrained Wide ResNet-50-2 backbone and a
PatchCore-style nearest-neighbor memory bank. Features from `layer2` and
`layer3` are combined on a 16x16 patch grid. Each test patch is scored by cosine
distance to its nearest normal training patch. The pixel map is smoothed and
resized to 256x256; the image score is the mean of the highest-scoring 1% of
pixels.

Run from the directory containing `data/` and `outputs/`:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

Training reads only `data/train/`. Inference reads only `data/test_images/`.
The pretrained torchvision weights may be downloaded automatically if they are
not already in the PyTorch cache.

Outputs:

- `outputs/image_scores.json`: raw nearest-neighbor anomaly scores (higher is
  more anomalous).
- `outputs/pixel_scores/*.png`: calibrated single-channel 16-bit anomaly maps.
- `outputs/model.pt`: learned normal patch memory bank and calibration values.

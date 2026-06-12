# One-Class Image Anomaly Detection

This solution uses a PatchCore-style detector with an ImageNet-pretrained
Wide ResNet-50-2 backbone. Intermediate layer-2 and layer-3 patch features are
locally averaged, concatenated, projected to 256 dimensions, and sampled into a
normal-feature memory bank. Inference uses nearest-neighbor distance to that
memory bank as the patch anomaly score.

## Run

Run these commands from the directory containing `data/` and `outputs/`:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The pretrained torchvision backbone weights must be available through the
normal torchvision cache/download mechanism. Training writes
`outputs/model.pt`. Inference writes `outputs/image_scores.json` and one
single-channel 256x256 PNG per test image under `outputs/pixel_scores/`.

Image scores are the mean of the highest 1% of smoothed patch distances.
Pixel PNGs are robustly scaled to 0-255 per image for storage and visualization;
larger values represent more anomalous pixels.

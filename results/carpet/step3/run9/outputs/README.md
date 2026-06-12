# Feature-Based Anomaly Detection

This solution uses a PatchCore-style nearest-neighbor detector. It extracts
localized `layer2` and `layer3` features from an ImageNet-pretrained Wide
ResNet-50-2, fuses them on a 32x32 grid, projects them to 384 dimensions, and
stores a sampled memory bank of normal training patches.

For inference, each test patch receives its cosine distance to the nearest
normal patch. The patch map is bilinearly resized and Gaussian-smoothed to
256x256. The image score is the mean of the highest-scoring 1% of pixels.

## Run

Run from the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The pretrained backbone weights may be downloaded by torchvision on first use.
The trained normal memory bank is saved as `outputs/model.pt`.

`pixel_scores/*.png` are single-channel 16-bit PNGs. A single shared linear
scale is used for all test maps; `score_metadata.json` records the multiplier
needed to recover approximate raw floating-point scores.

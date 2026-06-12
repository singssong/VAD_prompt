# Feature-Based Anomaly Detection

This solution uses a PatchCore-style nearest-neighbor detector. It extracts
local ImageNet-pretrained Wide ResNet-50-2 features from three backbone stages,
combines them at a 32x32 patch grid, applies a fixed random projection, and
stores a sampled memory bank of normal training patches. Pixel scores are the
distance to the nearest normal patch. The image score is the mean of the
highest-scoring 1% of pixels.

## Run

From the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first training run downloads the official torchvision ImageNet weights if
they are not already cached. CUDA is used automatically when available.

Generated files:

- `outputs/model.pt`: trained normal feature memory bank
- `outputs/image_scores.json`: one floating-point score per test image
- `outputs/pixel_scores/*.png`: 256x256 single-channel anomaly maps

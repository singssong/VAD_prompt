# Feature-Based Anomaly Detection

This solution uses a PatchCore-style normal feature memory. An ImageNet-pretrained
Wide ResNet-50-2 extracts and fuses intermediate layer-2 and layer-3 features on
a 16x16 patch grid. Test patches are scored by exact nearest-neighbor distance
to a sampled bank of standardized normal training patches.

Every input is resized to 256x256. Patch scores are bilinearly resized and
smoothed to produce the 256x256 pixel map. The image score is the mean of the
highest-scoring 1% of pixels.

## Run

Run from the task root (the directory containing `data/` and `outputs/`):

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The pretrained torchvision weights may be downloaded automatically if they are
not already in the PyTorch cache. CUDA is used automatically when available;
CPU execution is also supported. The trained normal memory is written to
`outputs/model.pt`.

Outputs:

- `outputs/image_scores.json`: raw floating-point anomaly score per test image.
- `outputs/pixel_scores/<test filename>`: single-channel 16-bit PNG anomaly map.

The PNG values use the fixed monotonic transform
`log(1 + score) / log(1 + 10000)`, clipped and scaled to `[0, 65535]`. Higher
values indicate greater anomaly.

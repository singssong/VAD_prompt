# Carpet anomaly detection

This solution uses a PatchCore-style normal feature memory. ImageNet-pretrained
Wide ResNet-50-2 features from `layer2` and `layer3` are pooled, fused, randomly
projected, and L2-normalized. Test patches are scored by nearest-neighbor
distance to the normal memory bank. Scores are robustly calibrated using a
held-out subset of normal training images.

## Run

From the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

`torchvision` downloads the ImageNet backbone weights on first use if they are
not already in the PyTorch cache. CUDA is used automatically when available.

Outputs:

- `outputs/image_scores.json`: one anomaly score per test filename
- `outputs/pixel_scores/`: single-channel 256x256 PNG anomaly maps
- `outputs/model.pt`: trained normal patch memory and calibration parameters

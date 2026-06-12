# Carpet anomaly detector

This solution uses a PatchCore-style one-class detector. It extracts multi-scale
patch features from an ImageNet-pretrained Wide ResNet-50-2, projects and
normalizes them, and stores a random representative memory bank of normal
training patches. Anomaly scores are nearest-neighbor cosine distances to that
normal memory.

Run from the task root:

```bash
python3 -m pip install -r outputs/requirements.txt
python3 outputs/train.py
python3 outputs/infer.py
```

The pretrained torchvision backbone weights must be available in the standard
PyTorch cache or downloadable on the first run. Training writes
`outputs/model.pt`. Inference writes `outputs/image_scores.json` and one
single-channel 256x256 PNG per test image under `outputs/pixel_scores/`.

Optional arguments:

```bash
python3 outputs/train.py --help
python3 outputs/infer.py --help
```

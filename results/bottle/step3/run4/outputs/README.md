# Feature-Based Anomaly Detection

This solution uses a PaDiM-style spatial feature-distribution model. It extracts
multi-scale features from an ImageNet-pretrained ResNet-18 (`layer1` through
`layer3`), selects a fixed 128-channel subspace, and estimates a diagonal
Gaussian distribution at every spatial location using only normal training
images. Pixel scores are diagonal Mahalanobis distances. Each image score is
the mean of the highest-scoring 1% of pixels.

All inputs are resized to 256x256 before feature extraction. Pixel maps are
saved as single-channel 16-bit PNG files at 256x256. Their values use one fixed
scale calibrated from normal training pixels, so values are comparable across
test images.

Run from the task directory:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first training run may download ImageNet ResNet-18 weights through
torchvision. Training writes `outputs/model.pt`; inference reads that local
checkpoint and writes:

- `outputs/image_scores.json`
- `outputs/pixel_scores/<test filename>.png`

Optional arguments are available with `python outputs/train.py --help` and
`python outputs/infer.py --help`.

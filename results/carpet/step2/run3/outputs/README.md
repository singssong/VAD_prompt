# Carpet anomaly detection

This solution uses a PatchCore-style one-class detector. An ImageNet-pretrained
ResNet-18 extracts intermediate `layer2` and `layer3` patch features. Training
stores a reproducible 50,000-patch coreset from normal images. Inference scores
each patch by its nearest normalized feature in that memory bank, smooths the
result into a 256x256 anomaly map, and uses the mean of the highest-scoring 1%
of pixels as the image anomaly score.

## Run

From the repository root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run downloads the torchvision ResNet-18 ImageNet weights. The scripts
default to `data/train`, `data/test_images`, and paths below `outputs`; command
line flags can override these paths. Inference creates `image_scores.json` and
one single-channel 256x256 PNG per test image under `pixel_scores/`.

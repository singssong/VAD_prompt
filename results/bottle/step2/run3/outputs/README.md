# Bottle Anomaly Detection

This solution uses a PatchCore-style patch memory bank built from normal
training images. Features come from the `layer2` and `layer3` stages of an
ImageNet-pretrained Wide ResNet-50-2. Each test patch is scored by cosine
distance to its nearest normal patch in a deterministic 256-channel subspace.

Run from the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

Training writes `outputs/model.pt`. Inference writes:

- `outputs/image_scores.json`
- `outputs/pixel_scores/<test filename>.png`

Image scores are the mean of the highest-scoring 1% of pixels. Pixel maps use
one dataset-wide unsupervised percentile scale and are saved as 8-bit,
single-channel 256x256 PNG files.

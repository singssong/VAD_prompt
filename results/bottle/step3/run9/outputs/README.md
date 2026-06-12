# Feature-Based Anomaly Detection

This solution uses a PatchCore-style normal feature memory. It extracts local
features from the `layer2` and `layer3` stages of an ImageNet-pretrained
Wide ResNet-50-2, projects the concatenated patch features to 256 dimensions,
and scores each test patch by its nearest-neighbor distance to normal training
patches.

All inputs are resized to 256x256. Patch distances are interpolated and smoothed
to make a 256x256 pixel anomaly map. The image score is the mean of the highest
1% of pixel scores.

Run from the project root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The training command writes `outputs/model.pt`. Inference writes
`outputs/image_scores.json` and one grayscale PNG per test image under
`outputs/pixel_scores/`. The first run may download the torchvision ImageNet
weights if they are not already cached.

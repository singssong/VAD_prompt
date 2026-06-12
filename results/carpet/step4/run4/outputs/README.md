# Feature-Based Anomaly Detection

This solution uses ImageNet-pretrained ResNet-50 features from `layer2` and
`layer3`. The two feature levels are resized to a common spatial grid,
channel-normalized, and concatenated. Training fits a diagonal Gaussian model
to normal spatial descriptors. Inference uses the resulting standardized
feature distance as a pixel anomaly map, applies Gaussian smoothing, and
aggregates the highest-scoring 1% of locations into an image score.

Image and pixel scores use a bounded monotonic normalization whose scale is a
robust high quantile measured only on the normal training images.

## Run

From the repository root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run may download the torchvision ImageNet weights if they are not
already cached. Inference writes `outputs/image_scores.json` and one grayscale
256x256 PNG per test image under `outputs/pixel_scores/`.

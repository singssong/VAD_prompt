# Bottle Anomaly Detection

This solution uses a PatchCore-style normal feature memory. An ImageNet-pretrained
Wide ResNet-50-2 extracts `layer2` and `layer3` feature maps. The maps are aligned,
concatenated, locally pooled, and compared with normal training patch features by
nearest-neighbor distance.

## Run

From the repository root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run may download the torchvision ImageNet weights. Training reads only
`data/train/`; inference reads only `data/test_images/`.

Inference writes:

- `outputs/image_scores.json`: min-max normalized image scores in `[0, 1]`
- `outputs/pixel_scores/`: globally normalized, single-channel 256x256 PNG maps

Pixel scores are nearest-normal-patch distances, Gaussian-smoothed at feature-map
resolution and then resized to 256x256. Image scores are the mean of the highest
1% of pixel scores.

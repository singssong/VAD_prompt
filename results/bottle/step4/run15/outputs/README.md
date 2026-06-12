# Feature-Based Anomaly Detection

This implementation uses a PatchCore-style normal patch memory bank. An
ImageNet-pretrained Wide ResNet-50-2 extracts `layer2` and `layer3` feature
maps, which are spatially aligned and concatenated. Test patch descriptors are
scored by distance to their nearest normal descriptor.

All inputs are resized to 256x256. Patch-distance maps are Gaussian-smoothed,
resized to 256x256, and saved as single-channel PNGs. The mean of the highest
1% of smoothed patch scores gives the image score. A held-out subset of normal
training images supplies robust calibration bounds used with a monotonic
exponential transform to normalize scores to the range `[0, 1]`.

## Run

From this `outputs` directory:

```bash
python -m pip install -r requirements.txt
python train.py
python infer.py
```

The pretrained torchvision backbone weights are downloaded automatically if
they are not already cached. Training writes `anomaly_model.pt`. Inference
writes `image_scores.json` and one grayscale 256x256 PNG per test image under
`pixel_scores/`.

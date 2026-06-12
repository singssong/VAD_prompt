# Feature-Based Anomaly Detection

This implementation uses a PatchCore-style nearest-neighbor feature memory. An
ImageNet-pretrained ResNet-18 extracts and concatenates `layer2` and `layer3`
features. A sampled bank of normal training patches models the normal feature
distribution. Test-patch distance to its nearest normal patch forms the anomaly
map.

All images are resized to 256x256. The low-resolution anomaly map is Gaussian
smoothed, bilinearly resized to 256x256, and robustly normalized using statistics
computed only from the normal training images. The image score is the mean of
the highest-scoring 1% of pixels.

## Run

From the directory containing `data/` and `outputs/`:

```bash
python3 -m pip install -r outputs/requirements.txt
python3 outputs/train.py
python3 outputs/infer.py
```

Training writes `outputs/model.pt`. Inference writes:

- `outputs/image_scores.json`
- `outputs/pixel_scores/<test filename>` for every test image

CUDA is used automatically when available; CPU execution is also supported.

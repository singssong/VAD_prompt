# Feature-based anomaly detection

This implementation uses an ImageNet-pretrained ResNet-18 and extracts aligned
mid-level features from `layer2` and `layer3`. The concatenated descriptors are
projected, L2-normalized, and compared with a memory bank sampled from normal
training patches. Nearest-neighbor cosine distance is the patch anomaly score.
Maps are Gaussian-smoothed and resized to 256x256; the mean of the highest 1%
of pixel scores is the image score. Final image and pixel scores are robustly
normalized to `[0, 1]`.

## Run

From this directory:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

Training reads only `data/train/`. Inference reads only `data/test_images/`.
The pretrained ResNet-18 weights may be downloaded by torchvision on first use.

Generated files:

- `outputs/model.pt`: fitted normal feature memory bank
- `outputs/image_scores.json`: normalized score for every test filename
- `outputs/pixel_scores/<test filename>`: single-channel 256x256 PNG map


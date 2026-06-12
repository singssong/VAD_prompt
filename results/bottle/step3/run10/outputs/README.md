# Feature-based anomaly detection

This implementation uses ImageNet-pretrained ResNet-50 features from `layer2`
and `layer3`. It projects and L2-normalizes local patch descriptors, stores a
random subset of normal training patches, and scores each test patch by cosine
distance to its nearest normal patch. The image score is the mean of the highest
1% of pixels in the smoothed anomaly map.

Run from the directory containing `data/` and `outputs/`:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run downloads torchvision's ImageNet ResNet-50 weights. Training
creates `outputs/model.pt`. Inference creates `outputs/image_scores.json` and
one 256x256 grayscale PNG per test image in `outputs/pixel_scores/`.

Optional arguments are documented with:

```bash
python outputs/train.py --help
python outputs/infer.py --help
```

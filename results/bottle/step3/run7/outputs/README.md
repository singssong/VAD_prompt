# Feature-Based Anomaly Detection

This solution fits a PaDiM-style spatial diagonal Gaussian to intermediate
ImageNet-pretrained Wide ResNet-50-2 features from normal training images.
Features from `layer2` and `layer3` are aligned on a 32x32 grid and projected
to 384 dimensions. At inference, standardized distance from the normal
distribution produces the pixel anomaly map. The image score is the mean of
the highest 1% of smoothed pixel scores.

All inputs are resized to 256x256. No labels or masks are used.

## Run

From the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run may download the official torchvision ImageNet weights. Training
creates `outputs/model.pt`. Inference creates `outputs/image_scores.json` and
one single-channel 256x256 PNG per test image under `outputs/pixel_scores/`.

CPU execution is supported with `--device cpu`; CUDA is selected automatically
when available.

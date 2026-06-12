# Anomaly Detection

This solution uses a PatchCore-style memory bank of multiscale ImageNet
ResNet-18 features. It trains only on normal images and requires no labels.

Run from the project root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first training run may download the standard torchvision ImageNet weights.
Training writes `outputs/model.pt`. Inference writes `outputs/image_scores.json`
and one 256x256 grayscale PNG per test image under `outputs/pixel_scores/`.

Higher JSON scores and brighter map pixels indicate greater anomaly evidence.

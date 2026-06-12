# Carpet anomaly detection

This solution uses a PatchCore-style one-class detector. ImageNet-pretrained
Wide ResNet-50-2 intermediate features are pooled into a 32x32 patch grid,
projected to 256 dimensions, and compared with a random coreset of normal
training patches using exact nearest-neighbor search.

Run from the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

Training reads only `data/train/`. Inference reads only `data/test_images/` and
the trained `outputs/model.pt`. It writes `outputs/image_scores.json` and one
256x256 grayscale PNG per test image under `outputs/pixel_scores/`.

The first run may download the standard torchvision
`Wide_ResNet50_2_Weights.IMAGENET1K_V2` checkpoint if it is not already cached.

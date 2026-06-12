# Carpet anomaly detection

This solution uses a PatchCore-style one-class detector with an
ImageNet-pretrained Wide ResNet-50-2 backbone. Intermediate `layer2` and
`layer3` patch features are pooled, randomly projected, and sampled into a
normal-feature memory bank. Test patch anomaly scores are nearest-neighbor
distances to that bank.

## Run

From the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

Training reads only `data/train/`. Inference reads only `data/test_images/` and
the generated `outputs/model.pt`.

Generated results:

- `outputs/image_scores.json`: one floating-point score per test filename
- `outputs/pixel_scores/`: one single-channel 256x256 PNG per test image

Higher image and pixel values indicate greater anomaly.

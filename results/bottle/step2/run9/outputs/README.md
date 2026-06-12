# Bottle Anomaly Detection

This solution uses a positional PatchCore-style detector with a frozen
ImageNet-pretrained Wide ResNet-50-2 backbone. Features from `layer2` and
`layer3` are fused on a 32x32 grid. Each test patch is scored by its nearest
normal training patch at the same spatial position.

## Run

From the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run may download torchvision's ImageNet Wide ResNet-50-2 weights if
they are not already in the PyTorch cache.

Training reads only `./data/train/`. Inference reads only
`./data/test_images/`. Generated results are:

- `outputs/image_scores.json`: one floating-point score per test filename
- `outputs/pixel_scores/*.png`: comparable grayscale anomaly maps, 256x256
- `outputs/model.pt`: normal feature memory produced by training

Higher image scores and brighter pixels indicate greater anomaly likelihood.

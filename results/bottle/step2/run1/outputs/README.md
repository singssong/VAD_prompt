# Bottle Anomaly Detection

This solution uses a PatchCore-style one-class detector with ImageNet
Wide ResNet-50-2 features. It fuses `layer1` and `layer2` features, projects
them to a compact embedding, and compares each test patch with a memory bank
of normal training patches.

Run from the task root:

```bash
python3 -m pip install -r outputs/requirements.txt
python3 outputs/train.py
python3 outputs/infer.py
```

Training reads only `data/train/`. Inference reads only `data/test_images/`
and the trained `outputs/model.pt` artifact. Pretrained ImageNet weights are
downloaded by torchvision if they are not already cached.

Generated results:

- `outputs/image_scores.json`: calibrated image anomaly scores; higher is more anomalous.
- `outputs/pixel_scores/<test filename>`: 256x256 single-channel PNG anomaly maps.

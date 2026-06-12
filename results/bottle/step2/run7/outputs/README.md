# Bottle Anomaly Detection

This solution uses a PatchCore-style one-class detector. It extracts local
ImageNet Wide ResNet-50-2 features from normal images, projects and samples
them into a compact memory bank, then scores each test patch by its nearest
normal neighbor.

Run from the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The pretrained backbone weights are downloaded by torchvision if they are not
already cached. Training reads only `data/train/`; inference reads only
`data/test_images/`. Results are written to `outputs/image_scores.json` and
`outputs/pixel_scores/`. The generated `outputs/model.pt` is the trained
normal-feature memory bank.

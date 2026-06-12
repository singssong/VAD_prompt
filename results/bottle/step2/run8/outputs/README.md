# Bottle Anomaly Detection

This solution uses a spatially aware PatchCore-style nearest-neighbor model.
ImageNet-pretrained ResNet-18 layer-2 and layer-3 features form multi-scale
patch descriptors. Normal training descriptors are sampled into a memory bank;
distance to the closest normal patch is the anomaly signal.

Run from the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

Training reads only `data/train/`. Inference reads only `data/test_images/` and
the trained `outputs/model.pt`. It writes `outputs/image_scores.json` and one
single-channel 256x256 PNG per test image under `outputs/pixel_scores/`.

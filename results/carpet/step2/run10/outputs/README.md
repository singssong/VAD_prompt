# Carpet Anomaly Detection

This solution uses a PatchCore-style nearest-neighbor memory bank built from
intermediate ImageNet-pretrained Wide ResNet-50-2 features. Only normal training
images are used to fit the feature normalization and memory bank.

Run from the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

Training writes `outputs/model.pt`. Inference writes `outputs/image_scores.json`
and one 256x256 grayscale PNG per test image under `outputs/pixel_scores/`.
Higher JSON scores and brighter map pixels indicate greater anomaly evidence.
Pixel maps use a shared cosine-distance scale across the complete test set.

The first run may download the standard torchvision ImageNet-1K V2 backbone
weights. CUDA is used automatically when available, with CPU fallback.

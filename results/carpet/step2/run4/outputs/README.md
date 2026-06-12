# Carpet anomaly detection

This solution uses a PatchCore-style one-class detector with ImageNet-pretrained
Wide ResNet-50-2 intermediate features. Training stores a random representative
memory bank of normal carpet patches. Inference uses exact nearest-neighbor
distance to that bank for image and pixel anomaly scores.
Position-wise calibration on reserved normal training images removes systematic
feature-map border bias.

Run from the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first training run downloads the torchvision
`Wide_ResNet50_2_Weights.IMAGENET1K_V2` weights. CUDA is used automatically when
available; CPU execution is supported but slower.

Outputs:

- `outputs/model.pt`: learned normal patch memory bank and calibration values
- `outputs/image_scores.json`: anomaly score for every test image
- `outputs/pixel_scores/*.png`: 256x256 single-channel anomaly maps

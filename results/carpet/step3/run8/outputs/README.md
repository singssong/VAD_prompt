# Feature-based anomaly detection

This solution uses frozen ImageNet-pretrained ResNet-18 features from `layer1`
and `layer2`. The multi-scale feature maps are aligned into a 32x32 patch grid
and L2-normalized. Training fits 256 spherical k-means prototypes to normal
patch descriptors. At inference, each patch is scored by its cosine distance
to the nearest normal prototype.

The 32x32 anomaly grid is smoothed and resized to 256x256. The image score is
the mean of the highest-scoring 1% of pixels. Pixel PNG values use a fixed
scale estimated solely from normal training distances.

Run from the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first training run may download the official torchvision ResNet-18
ImageNet weights. Inference uses the backbone weights stored in `model.pt`.


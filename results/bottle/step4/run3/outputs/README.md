# Feature-Based Anomaly Detection

This solution uses ImageNet-pretrained Wide ResNet-50-2 mid-level features from
`layer2` and `layer3`. The feature maps are resized to a common 32x32 grid,
concatenated, randomly projected, and L2-normalized. Each test patch is scored by
its nearest normal training descriptor at the same spatial location.

The raw 32x32 anomaly map is Gaussian-smoothed and resized to 256x256. The image
score is the mean of the highest-scoring 1% of map pixels. Both image and pixel
scores are normalized using leave-one-out statistics computed only from normal
training images.

## Run

From the repository root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The pretrained `Wide_ResNet50_2_Weights.IMAGENET1K_V2` weights are downloaded
by torchvision on first use if they are not already cached.

Inference writes:

- `outputs/image_scores.json`
- `outputs/pixel_scores/<test filename>`

The model checkpoint is stored as `outputs/model.pt`.

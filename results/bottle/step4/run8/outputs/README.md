# Feature-Based Anomaly Detection

This solution uses a PatchCore-style nearest-neighbor detector. It extracts
ImageNet-pretrained Wide ResNet-50-2 features from `layer2` and `layer3`,
concatenates and projects the spatial patch descriptors, and stores a sampled
memory bank of normal training patches. Test patch scores are distances to the
nearest normal patch.

The pixel map is Gaussian-smoothed at feature resolution and resized to
256x256. The image score is the mean of the highest-scoring 1% of patches.
Both image and pixel scores are normalized using held-out normal training
images. No test labels or masks are used.

## Run

From this directory:

```bash
python -m pip install -r requirements.txt
python train.py
python infer.py
```

The pretrained backbone weights are downloaded by torchvision on first use.
Inference writes `image_scores.json` and one grayscale 256x256 PNG per test
image under `pixel_scores/`.

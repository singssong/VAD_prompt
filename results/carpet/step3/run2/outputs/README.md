# Feature-Based Anomaly Detection

This solution uses an ImageNet-pretrained Wide ResNet-50-2 as a frozen feature
extractor. Intermediate layer-2 and layer-3 features are fused into a 32x32
patch grid, randomly projected, and L2-normalized. Training samples a memory
bank of normal patch descriptors. Inference uses cosine nearest-neighbor
distance to the normal memory bank as the pixel anomaly signal. The image score
is the mean of the highest-scoring 1% of pixels.

All inputs are resized to 256x256 before feature extraction. Pixel maps are
bilinearly resized and Gaussian-smoothed to 256x256.

## Run

From the project root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

CUDA is selected automatically when available. To force CPU execution:

```bash
python outputs/train.py --device cpu
python outputs/infer.py --device cpu
```

The pretrained torchvision weights may be downloaded on first use if they are
not already in the PyTorch cache. Training writes `outputs/model.pt`.
Inference writes `outputs/image_scores.json` and one grayscale 256x256 PNG per
test image under `outputs/pixel_scores/`.

# Feature-Based Bottle Anomaly Detection

This pipeline uses an ImageNet-pretrained ResNet-18 and concatenates spatially
aligned features from `layer1`, `layer2`, and `layer3`. Training fits a diagonal
Gaussian to normal features at each spatial position. Inference computes a
standardized feature distance, Gaussian-smooths the pixel map, resizes it to
256x256, and averages the highest-scoring 1% of pixels for the image score.
Scores are mapped to `[0, 1)` using scales estimated only from training images.

## Run

From this directory:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The scripts read only `data/train/` and `data/test_images/`. Generated results
are written to:

- `outputs/image_scores.json`
- `outputs/pixel_scores/`
- `outputs/normal_model.pt`

The first run may download the official torchvision ResNet-18 ImageNet weights
if they are not already in the PyTorch cache. CUDA is used automatically when
available.

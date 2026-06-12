# Feature-based anomaly detection

This implementation uses ImageNet-pretrained ResNet-18 features from `layer1`
and `layer2`. The aligned feature maps are concatenated and compared with a
sampled memory bank of normal training patches using nearest-neighbor distance.
A held-out subset of normal training images calibrates both pixel and image
scores. Pixel maps are Gaussian-smoothed before resizing to 256x256, and image
scores aggregate the highest-scoring 1% of pixels.

## Run

From this directory:

```bash
python -m pip install -r requirements.txt
python train.py
python infer.py
```

The scripts resolve `../data/train` and `../data/test_images` relative to the
`outputs` directory. Inference writes `image_scores.json` and one grayscale PNG
per test image under `pixel_scores/`.

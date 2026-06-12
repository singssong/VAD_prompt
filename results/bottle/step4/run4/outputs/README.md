# Feature-Based Anomaly Detection

This implementation uses ImageNet-pretrained Wide ResNet-50-2 features from
`layer1` and `layer2`. The features are concatenated on a shared spatial grid,
and a position-aware diagonal Gaussian is fitted from normal training images.
Inference uses per-location standardized feature distance. Pixel maps are
Gaussian-smoothed and resized to 256x256; image scores are the mean of the
highest-scoring 1% of pixels and are monotonically normalized to `[0, 1)`.

## Run

From this directory:

```bash
python -m pip install -r requirements.txt
python train.py
python infer.py
```

The pretrained torchvision weights are downloaded automatically if they are
not already in the PyTorch cache. The scripts default to `../data/train` and
`../data/test_images`, and write the model and requested results here.

# Feature-Based Anomaly Detection

This solution uses an ImageNet-pretrained ResNet-18. Features from `layer1`,
`layer2`, and `layer3` are resized to a common 64x64 grid and concatenated. A
position-aware diagonal Gaussian is fitted using only normal training features.
Inference uses diagonal Mahalanobis distance, Gaussian smoothing, and the mean
of the highest-scoring 1% of pixels as the image anomaly score.

Image and pixel scores use a bounded monotonic normalization calibrated with
robust quantiles measured on the normal training set. JSON image scores are
floats in `[0, 1]`, where higher is more anomalous. Pixel maps are 8-bit,
single-channel 256x256 PNG files.

## Run

From this directory:

```bash
python -m pip install -r requirements.txt
python train.py
python infer.py
```

The pretrained ResNet-18 weights are downloaded by torchvision if they are not
already in the PyTorch cache. Training reads only `../data/train/`; inference
reads only `../data/test_images/`.

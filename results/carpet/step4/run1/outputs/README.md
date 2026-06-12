# Feature-Based Anomaly Detection

This pipeline uses an ImageNet-pretrained ResNet-18 and a PatchCore-style
normal patch memory bank. Features from `layer2` and `layer3` are resized to a
common grid, concatenated, projected, and compared with normal training
features using nearest-neighbor distance.

The pixel anomaly map is Gaussian-smoothed before it is resized to 256x256.
The image score is the mean of the highest-scoring 1% of map locations.
Pixel and image scores are normalized using a held-out subset of normal
training images. Image scores use a bounded monotonic transform so unusually
large anomaly distances remain ranked instead of being hard-clipped.

## Run

From this directory:

```bash
python -m pip install -r requirements.txt
python train.py
python infer.py
```

The scripts resolve `../data/train` and `../data/test_images` relative to this
directory. `train.py` writes `normal_model.pt`; `infer.py` writes
`image_scores.json` and one grayscale PNG per test image under `pixel_scores/`.
An NVIDIA GPU is used automatically when available, with CPU fallback.

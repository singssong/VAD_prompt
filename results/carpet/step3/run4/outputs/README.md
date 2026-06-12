# Feature-Based Anomaly Detection

This solution uses a PatchCore-style normal patch memory bank. It extracts
ImageNet-pretrained Wide ResNet-50-2 `layer2` and `layer3` features on a 32x32
spatial grid, projects and normalizes them, and scores each test patch by cosine
distance to its nearest normal training patch.

The pixel map is bilinearly resized and Gaussian-smoothed to 256x256. The image
score is the mean of the highest-scoring 1% of pixels. Pixel PNG values are
16-bit grayscale scores calibrated with a held-out subset of normal training
images; larger values indicate greater anomaly.

## Run

Run from the task root (the directory containing `data/` and `outputs/`):

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The pretrained backbone weights are downloaded by torchvision if they are not
already cached. Training writes `outputs/model.pt`. Inference writes
`outputs/image_scores.json` and one 256x256 single-channel PNG per test image
under `outputs/pixel_scores/`.

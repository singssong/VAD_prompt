# Carpet anomaly detection

This solution uses PatchCore with an ImageNet-pretrained Wide ResNet-50-2
backbone. Features from the first three residual stages are pooled to a 32x32
grid, locally averaged, reduced to a deterministic 384-channel embedding, and
stored in a 40,000-patch random coreset. Pixel scores are nearest-neighbor cosine
distances to this normal memory bank; image scores are the mean of the
highest-scoring 1% of pixels.

## Run

From the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The pretrained torchvision weights are downloaded automatically if they are not
already cached. Training reads `./data/train/`; inference reads
`./data/test_images/`. The fitted model is saved as `./outputs/model.pt`.

Inference writes:

- `./outputs/image_scores.json`
- `./outputs/pixel_scores/<test filename>.png`

All pixel maps are single-channel 8-bit PNG images at 256x256 resolution. Higher
image and pixel values indicate greater anomaly likelihood.

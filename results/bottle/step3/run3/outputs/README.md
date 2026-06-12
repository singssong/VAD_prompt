# Feature-Based Anomaly Detection

This solution uses ImageNet-pretrained Wide ResNet-50-2 intermediate features and
a PaDiM-style positional diagonal Gaussian model. Each patch is scored by its
standardized distance from normal training features. The pixel map is bilinearly
upsampled and smoothed to 256x256; the image score is the mean of the highest 1%
of pixel scores.

## Run

From the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run may download the torchvision ImageNet weights. Training creates
`outputs/model.pt`. Inference creates `outputs/image_scores.json` and one
single-channel 256x256 PNG per test image under `outputs/pixel_scores/`.

Optional arguments:

```bash
python outputs/train.py --device cpu --batch-size 4
python outputs/infer.py --device cpu --batch-size 4
```

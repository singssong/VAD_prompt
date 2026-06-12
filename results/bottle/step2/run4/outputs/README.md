# Bottle Anomaly Detection

This solution fits a PaDiM-inspired spatial diagonal Gaussian model to
multi-scale features from an ImageNet-pretrained Wide ResNet-50-2. Only normal
images from `data/train` are used to estimate the feature distribution,
foreground mask, and output calibration.

From the task root, install dependencies and run:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The torchvision backbone weights download automatically on first use if they
are not already in the PyTorch cache. CUDA is used when available; CPU also
works. The inference command writes `outputs/image_scores.json` and one
single-channel 256x256 PNG per test image under `outputs/pixel_scores`.

Optional arguments:

```bash
python outputs/train.py --train-dir ./data/train --model-out ./outputs/model.pt
python outputs/infer.py --test-dir ./data/test_images --model ./outputs/model.pt
```

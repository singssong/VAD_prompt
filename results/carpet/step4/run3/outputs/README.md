# Feature-Based Anomaly Detection

This solution uses a PatchCore-style normal feature memory. ImageNet-pretrained
Wide ResNet-50-2 features from `layer2` and `layer3` are aligned, concatenated,
randomly projected, and compared with normal training patches using cosine
nearest-neighbor distance. Pixel maps are Gaussian-smoothed and image scores
are top-1% map averages normalized by held-out normal calibration images.

## Run

From the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

Training reads only `data/train/`. Inference reads only `data/test_images/`.
The scripts resize every image to 256x256 and automatically use CUDA when
available. The pretrained torchvision backbone weights may be downloaded on
first use if they are not already cached.

Generated artifacts:

- `outputs/normal_feature_model.pt`
- `outputs/image_scores.json`
- `outputs/pixel_scores/<test filename>`

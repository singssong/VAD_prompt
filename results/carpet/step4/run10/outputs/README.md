# Feature-based anomaly detection

This implementation uses an ImageNet-pretrained Wide ResNet-50-2 backbone and
concatenates projected features from `layer2` and `layer3`. A memory bank of
normal patch features is built from the training images. At inference, each
test patch is scored by cosine distance to its nearest normal patch.

The low-resolution patch-distance map is Gaussian-smoothed, normalized using
statistics measured on normal training images, and resized to 256x256. The
image score is the normalized mean of the highest-scoring 1% of map locations.

## Run

From this `outputs` directory:

```bash
python -m pip install -r requirements.txt
python train.py
python infer.py
```

The scripts read only `../data/train/` for model fitting and
`../data/test_images/` for inference. Training writes `normal_model.pt`.
Inference writes `image_scores.json` and one single-channel 256x256 PNG per
test image under `pixel_scores/`.

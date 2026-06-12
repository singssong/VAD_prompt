# Feature-based anomaly detection

This solution uses ImageNet-pretrained Wide ResNet-50-2 features from layers 2
and 3. Training samples a memory bank of normal patch descriptors. Inference
scores each test patch by its Euclidean distance to the nearest normal patch,
smooths and resizes that map to 256x256, and uses the mean of the highest 1% of
pixel scores as the image anomaly score.

## Run

From the task directory:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The pretrained torchvision weights are downloaded automatically if they are not
already cached. The trained memory bank is written to `outputs/model.pt`.
`image_scores.json` contains raw comparable distances (higher is more
anomalous). Pixel maps are single-channel 16-bit PNGs, globally scaled over the
test set for consistent visualization.

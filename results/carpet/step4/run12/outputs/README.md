# Feature-Based Anomaly Detection

This implementation uses a frozen ImageNet-pretrained ResNet-18. It concatenates
spatial features from `layer2` and `layer3`, then models normality with a sampled
memory bank of normal patch descriptors. Test patch anomaly scores are nearest
neighbor distances to that bank. The pixel map is Gaussian-smoothed and resized
to 256x256; the image score is the mean of the highest-scoring 1% of pixels.
Scores are robustly normalized to `[0, 1]`.

## Run

From this directory's parent (`run12`):

```bash
python3 -m pip install -r outputs/requirements.txt
python3 outputs/train.py
python3 outputs/infer.py
```

The first run may download the official torchvision ImageNet ResNet-18 weights.
Training reads only `data/train/`; inference reads only `data/test_images/`.

Generated artifacts:

- `outputs/normal_memory.pt`
- `outputs/image_scores.json`
- `outputs/pixel_scores/<test filename>`

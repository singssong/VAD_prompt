# Feature-Based Anomaly Detection

This solution uses a frozen ImageNet-pretrained ResNet-18. Features from
`layer1`, `layer2`, and `layer3` are aligned to a 32x32 grid, concatenated,
randomly projected, and L2-normalized. Training samples a memory bank of normal
patch features. Inference assigns each patch its distance to the nearest normal
memory feature, upsamples the resulting map to 256x256, and averages the top 1%
of pixel distances for the image score.

## Run

From the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The training script reads only `./data/train/`. The inference script reads only
`./data/test_images/` and `./outputs/model.pt`. Results are written to
`./outputs/image_scores.json` and `./outputs/pixel_scores/`.

CUDA is used automatically when available. Pass `--device cpu` to either script
to force CPU execution.

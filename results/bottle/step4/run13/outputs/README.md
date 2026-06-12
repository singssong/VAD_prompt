# Feature-Based Anomaly Detection

This solution uses an ImageNet-pretrained ResNet-18 and concatenates its
`layer2` and upsampled `layer3` feature grids. A sampled memory bank models
normal patch features. Test patches are scored by cosine distance to their
nearest normal patch (a PatchCore-style approach).

Maps are Gaussian-smoothed at feature resolution, bilinearly resized to
256x256, and calibrated to 8-bit single-channel PNGs. Image scores are the mean
of the highest-scoring 1% of map locations and are mapped consistently to
`[0, 1]` with logistic calibration from a held-out subset of normal images.

## Run

From the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

Optional path overrides:

```bash
python outputs/train.py --train-dir data/train --model-path outputs/model.pt
python outputs/infer.py --test-dir data/test_images \
  --model-path outputs/model.pt --output-dir outputs
```

The first run may download the torchvision ResNet-18 ImageNet weights if they
are not already cached.

Outputs:

- `outputs/image_scores.json`: normalized image anomaly scores
- `outputs/pixel_scores/<test filename>`: 256x256 grayscale anomaly maps
- `outputs/model.pt`: trained normal-feature memory and calibration

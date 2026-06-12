# Feature-Based Anomaly Detection

This solution uses a PatchCore-style feature memory bank. An ImageNet-pretrained
Wide ResNet-50-2 extracts and concatenates normalized `layer2` and `layer3`
features on a 32x32 patch grid. Training stores a deterministic sample of normal
patch embeddings. Inference assigns each test patch its nearest-memory cosine
distance, interpolates and smooths the patch scores to 256x256, and uses the mean
of the highest-scoring 1% of pixels as the image anomaly score.

Run from the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The pretrained weights are downloaded by torchvision if they are not already in
the PyTorch cache. CUDA is used automatically when available. Optional paths and
batch sizes are documented by `python outputs/train.py --help` and
`python outputs/infer.py --help`.

Generated files:

- `outputs/model.pt`: learned normal feature memory bank
- `outputs/image_scores.json`: one floating-point score per test filename
- `outputs/pixel_scores/*.png`: fixed-scale, 16-bit, single-channel 256x256 maps

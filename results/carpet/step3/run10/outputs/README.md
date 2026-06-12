# Feature-Based Anomaly Detection

This solution uses ImageNet-pretrained Wide ResNet-50-2 features from `layer2`
and `layer3`. Training samples local patch embeddings from normal images to
form a memory bank. Inference assigns each patch its nearest-neighbor cosine
distance from that bank, upsamples and smooths the patch map to 256x256, and
uses the mean of the highest-scoring 1% of pixels as the image score.

Run from the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run downloads the torchvision ImageNet weights if they are not
already cached. CUDA is used automatically when available. The generated
pixel maps are single-channel 16-bit PNGs on a common `[0, 1]` cosine-distance
scale, encoded as `[0, 65535]`.

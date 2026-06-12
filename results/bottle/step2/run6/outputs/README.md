# Bottle anomaly detection

This solution uses PaDiM with an ImageNet-pretrained Wide ResNet-50-2 backbone.
Intermediate features are sampled to 100 dimensions and a regularized
multivariate Gaussian is fitted independently at every spatial position using
only normal training images. Mahalanobis distance produces the anomaly map.

Run from the task root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

Training writes `outputs/model.pt`. Inference writes `outputs/image_scores.json`
and one 8-bit, single-channel, 256x256 PNG per test image under
`outputs/pixel_scores/`. Image scores are robustly normalized against normal
training scores; larger values are more anomalous.

# Feature-based anomaly detection

This solution uses an ImageNet-pretrained ResNet-18. It concatenates normalized
features from `layer1` and `layer2`, models the normal feature distribution at
each spatial position with a diagonal Gaussian, and scores pixels using
Mahalanobis distance. Maps are Gaussian-smoothed before resizing to 256x256.
Image scores are the mean of the highest-scoring 1% of map locations and are
normalized to `[0, 1]` with a bounded monotonic transform whose scale is learned
only from training-image quantiles.

## Run

From the project root:

```bash
python -m pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

Training reads only `data/train/`. Inference reads only `data/test_images/`.
The trained artifact is saved as `outputs/normal_model.pt`.

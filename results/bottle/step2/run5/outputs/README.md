# Bottle Anomaly Detection

This solution uses PaDiM with ImageNet-pretrained Wide ResNet-50-2 features. It
fits a regularized multivariate Gaussian to normal feature vectors at each
spatial location. Mahalanobis distance provides both image anomaly scores and
localized pixel anomaly maps.

Run from the project root:

```bash
pip install -r outputs/requirements.txt
python outputs/train.py
python outputs/infer.py
```

The first run may download the standard torchvision
`Wide_ResNet50_2_Weights.IMAGENET1K_V2` weights. Training reads only
`data/train/`; inference reads only `data/test_images/`. The generated files are
`outputs/model.pt`, `outputs/image_scores.json`, and one 256x256 grayscale PNG
per test image under `outputs/pixel_scores/`.

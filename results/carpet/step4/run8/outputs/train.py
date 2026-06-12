import torch

import config
from pipeline import build_feature_extractor, calibrate_model, fit_normal_model, make_loader


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_loader(config.TRAIN_DIR)
    extractor = build_feature_extractor(device)

    normal_model = fit_normal_model(extractor, loader, device)
    normal_model = calibrate_model(extractor, loader, normal_model, device)
    normal_model["backbone"] = config.BACKBONE
    normal_model["feature_layers"] = config.FEATURE_LAYERS
    normal_model["image_size"] = config.IMAGE_SIZE
    torch.save(normal_model, config.MODEL_PATH)
    print(f"Saved normal feature model to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()

import torch

from config import MODEL_PATH, TRAIN_DIR
from pipeline import (
    build_feature_extractor,
    calibrate_normal_scores,
    fit_normal_model,
    list_images,
    make_loader,
    seed_everything,
)


def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = list_images(TRAIN_DIR)
    if not paths:
        raise RuntimeError(f"No training images found in {TRAIN_DIR}")

    extractor = build_feature_extractor(device)
    loader = make_loader(paths)
    model = fit_normal_model(extractor, loader, device)
    model = calibrate_normal_scores(extractor, loader, model, device)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, MODEL_PATH)
    print(f"Saved normal-feature model to {MODEL_PATH} using {len(paths)} images.")


if __name__ == "__main__":
    main()

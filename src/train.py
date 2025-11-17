from __future__ import annotations

import os
from ultralytics import YOLO

from utils import load_config


def main():
    cfg = load_config("configs/config.yaml")
    data_yaml = os.path.join("configs", "dataset.yolov8.yaml")
    model_name = cfg["training"]["model"]
    epochs = int(cfg["training"]["epochs"])  # type: ignore
    imgsz = int(cfg["training"]["imgsz"])  # type: ignore
    batch = int(cfg["training"]["batch"])  # type: ignore
    workers = int(cfg["training"]["workers"])  # type: ignore
    device = cfg["training"]["device"]
    runs_dir = cfg["paths"]["runs_dir"]
    os.makedirs(runs_dir, exist_ok=True)

    model = YOLO(model_name)
    # Optional learning rate from config for fine-tuning
    lr = cfg["training"].get("learning_rate") if "learning_rate" in cfg.get("training", {}) else None
    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=runs_dir,
        name="yolov8",
    )
    if lr is not None:
        train_kwargs["lr"] = float(lr)

    results = model.train(**train_kwargs)
    # Save best weights path to a file for later inference
    best_path = model.ckpt_path if hasattr(model, "ckpt_path") else None
    if best_path is None:
        # ultralytics exposes model.ckpt_path in recent versions; fall back to search
        exp_dir = results.save_dir  # type: ignore[attr-defined]
        candidate = os.path.join(exp_dir, "weights", "best.pt")
        if os.path.isfile(candidate):
            best_path = candidate
    if best_path:
        with open(os.path.join(cfg["paths"]["models_dir"], "best_model.path"), "w", encoding="utf-8") as f:
            f.write(best_path)
        print(f"Saved best model path: {best_path}")


if __name__ == "__main__":
    main()




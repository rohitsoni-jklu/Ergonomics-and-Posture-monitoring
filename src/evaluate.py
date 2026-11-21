"""Step 6: compare Model 1 and Model 2 on the SAME unseen test images.
Saves results/comparison.csv, confusion matrices, reports and models/best_model.json."""
import json
import time
import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from tensorflow import keras

from config import (ROOT, FEATURES_CSV, FEATURE_NAMES, CLASS_NAMES, IMG_SIZE,
                    MODELS_DIR, RESULTS_DIR)
from plot_utils import plot_confusion
from pose_utils import PoseFeatureExtractor

N_TIMING = 100


def load_rgb(path):
    img = cv2.imread(str(ROOT / path))
    return img


def metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision_macro": p,
            "recall_macro": r, "f1_macro": f1}


def main():
    if not FEATURES_CSV.exists():
        raise SystemExit("Run extract_features.py first.")
    df = pd.read_csv(FEATURES_CSV)
    test = df[df.split == "test"].reset_index(drop=True)
    if test.empty:
        raise SystemExit("The test split is empty.")
    y_true = test["label"].map({c: i for i, c in enumerate(CLASS_NAMES)}).values
    labels = list(range(len(CLASS_NAMES)))
    print(f"Test images: {len(test)} (from people not used for training)\n")

    # ---- Model 1 (features from the CSV) ----
    m1 = keras.models.load_model(MODELS_DIR / "model1_mlp.keras")
    scaler = joblib.load(MODELS_DIR / "model1_scaler.joblib")
    p1 = m1.predict(scaler.transform(test[FEATURE_NAMES]), verbose=0).argmax(1)

    # ---- Model 2 (images) ----
    m2 = keras.models.load_model(MODELS_DIR / "model2_cnn.keras")

    def prep(path):
        img = tf.io.decode_image(tf.io.read_file(str(ROOT / path)), channels=3, expand_animations=False)
        return tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy()

    X2 = np.stack([prep(p) for p in test["filepath"]])
    p2 = m2.predict(X2, batch_size=32, verbose=0).argmax(1)

    # ---- Speed: full per-frame pipeline, one image at a time ----
    sample = test["filepath"].tolist()[:N_TIMING]
    ext = PoseFeatureExtractor(static_image_mode=True)
    t0 = time.perf_counter()
    for p in sample:
        f, _ = ext.extract(load_rgb(p))
        if f is not None:
            m1(scaler.transform(f.reshape(1, -1)), training=False)
    ms1 = (time.perf_counter() - t0) / len(sample) * 1000
    ext.close()

    t0 = time.perf_counter()
    for p in sample:
        img = cv2.cvtColor(load_rgb(p), cv2.COLOR_BGR2RGB)
        x = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32")[None]
        m2(x, training=False)
    ms2 = (time.perf_counter() - t0) / len(sample) * 1000

    rows = []
    for name, pred, ms in (("Model 1 (Keypoint MLP)", p1, ms1), ("Model 2 (MobileNetV2 CNN)", p2, ms2)):
        m = metrics(y_true, pred)
        m.update({"model": name, "ms_per_frame": round(ms, 1), "fps": round(1000 / ms, 1)})
        rows.append(m)
        tag = "model1" if name.startswith("Model 1") else "model2"
        cm = confusion_matrix(y_true, pred, labels=labels)
        plot_confusion(cm, CLASS_NAMES, RESULTS_DIR / f"confusion_{tag}.png", name)
        (RESULTS_DIR / f"report_{tag}.txt").write_text(
            classification_report(y_true, pred, labels=labels, target_names=CLASS_NAMES, zero_division=0))

    res = pd.DataFrame(rows)[["model", "accuracy", "precision_macro", "recall_macro",
                              "f1_macro", "ms_per_frame", "fps"]].round(3)
    res.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    print(res.to_string(index=False))

    best = "model1" if rows[0]["f1_macro"] >= rows[1]["f1_macro"] else "model2"
    (MODELS_DIR / "best_model.json").write_text(json.dumps({"best": best}))
    print(f"\nBest model by macro F1: {best}  (saved to models/best_model.json)")
    print("Confusion matrices and reports are in the results/ folder.")


if __name__ == "__main__":
    main()

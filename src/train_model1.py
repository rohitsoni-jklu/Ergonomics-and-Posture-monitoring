"""Step 4: Model 1 - Keypoint-based MLP.
Input: 8 pose features (angles/ratios).  Output: posture class."""
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from config import FEATURES_CSV, FEATURE_NAMES, CLASS_NAMES, MODELS_DIR, RESULTS_DIR, SEED
from plot_utils import plot_history


def main():
    tf.keras.utils.set_random_seed(SEED)
    if not FEATURES_CSV.exists():
        raise SystemExit("data/features.csv not found. Run extract_features.py first.")
    df = pd.read_csv(FEATURES_CSV)
    df["y"] = df["label"].map({c: i for i, c in enumerate(CLASS_NAMES)})

    tr, va = df[df.split == "train"], df[df.split == "val"]
    if tr.empty or va.empty:
        raise SystemExit("Train or validation split is empty. Check data/splits.csv.")

    scaler = StandardScaler().fit(tr[FEATURE_NAMES])
    Xtr, Xva = scaler.transform(tr[FEATURE_NAMES]), scaler.transform(va[FEATURE_NAMES])
    ytr, yva = tr["y"].values, va["y"].values

    present = np.unique(ytr)
    cw = compute_class_weight("balanced", classes=present, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(present, cw)}

    model = keras.Sequential([
        layers.Input(shape=(len(FEATURE_NAMES),)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ], name="model1_keypoint_mlp")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    hist = model.fit(
        Xtr, ytr, validation_data=(Xva, yva), epochs=150, batch_size=32,
        class_weight=class_weight, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                                 restore_best_weights=True)])

    model.save(MODELS_DIR / "model1_mlp.keras")
    joblib.dump(scaler, MODELS_DIR / "model1_scaler.joblib")
    (MODELS_DIR / "class_names.json").write_text(json.dumps(CLASS_NAMES))
    plot_history(hist, RESULTS_DIR / "model1_training.png", "Model 1 (MLP)")
    print("\nSaved models/model1_mlp.keras and models/model1_scaler.joblib")
    print(f"Best validation accuracy: {max(hist.history['val_accuracy']):.3f}")


if __name__ == "__main__":
    main()

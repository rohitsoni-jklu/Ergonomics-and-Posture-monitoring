"""Step 5: Model 2 - Image-based CNN (MobileNetV2 transfer learning).
Input: webcam image (224x224).  Output: posture class.
First run downloads the ImageNet weights (needs internet once)."""
import json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from config import ROOT, CLASS_NAMES, IMG_SIZE, MODELS_DIR, RESULTS_DIR, SEED
from plot_utils import plot_history
from split_utils import load_splits

BATCH = 32


def make_ds(df, training):
    paths = [str(ROOT / p) for p in df["filepath"]]
    labels = df["label"].map({c: i for i, c in enumerate(CLASS_NAMES)}).values.astype("int32")

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model():
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.RandomBrightness(0.2)(inp)                 # augmentation (training only)
    x = layers.RandomContrast(0.2)(x)
    x = layers.RandomRotation(0.04)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)         # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return keras.Model(inp, out, name="model2_mobilenetv2"), base


def main():
    tf.keras.utils.set_random_seed(SEED)
    df = load_splits()
    tr, va = df[df.split == "train"], df[df.split == "val"]
    if tr.empty or va.empty:
        raise SystemExit("Train or validation split is empty. Check data/splits.csv.")

    ytr = tr["label"].map({c: i for i, c in enumerate(CLASS_NAMES)}).values
    present = np.unique(ytr)
    cw = compute_class_weight("balanced", classes=present, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(present, cw)}

    train_ds, val_ds = make_ds(tr, True), make_ds(va, False)
    model, base = build_model()
    stop = lambda: keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                 restore_best_weights=True)

    # Phase 1: train only the new top layers
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=12,
                   class_weight=class_weight, callbacks=[stop()], verbose=2)

    # Phase 2: fine-tune the last 30 layers of MobileNetV2 with a small learning rate
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=12,
                   class_weight=class_weight, callbacks=[stop()], verbose=2)

    merged = {k: h1.history[k] + h2.history[k] for k in h1.history}
    plot_history(merged, RESULTS_DIR / "model2_training.png", "Model 2 (CNN)")
    model.save(MODELS_DIR / "model2_cnn.keras")
    (MODELS_DIR / "class_names.json").write_text(json.dumps(CLASS_NAMES))
    print("\nSaved models/model2_cnn.keras")
    print(f"Best validation accuracy: {max(merged['val_accuracy']):.3f}")


if __name__ == "__main__":
    main()

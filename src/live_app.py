"""Step 7: real-time posture monitoring with alerts and a session report.

Usage:   python src/live_app.py                 (uses the best model from evaluate.py)
         python src/live_app.py --model model1  (or model2)
Press q to quit. A report is saved to results/ when you quit.
"""
import argparse
import json
import time
from collections import deque
from datetime import datetime

import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (CLASS_NAMES, GOOD_CLASS, IMG_SIZE, MODELS_DIR, RESULTS_DIR, CAMERA_INDEX,
                    MIRROR, SMOOTH_WINDOW, ALERT_SECONDS, ALERT_COOLDOWN)


def notify(msg):
    try:
        from plyer import notification
        notification.notify(title="Posture alert", message=msg, timeout=5)
    except Exception:
        print("\a" + msg)


def choose_model(arg):
    if arg:
        return arg
    f = MODELS_DIR / "best_model.json"
    if f.exists():
        return json.loads(f.read_text())["best"]
    print("best_model.json not found (run evaluate.py). Using model1.")
    return "model1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["model1", "model2"])
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX)
    ap.add_argument("--alert_seconds", type=float, default=ALERT_SECONDS)
    args = ap.parse_args()
    which = choose_model(args.model)

    from tensorflow import keras
    extractor = scaler = None
    if which == "model1":
        from pose_utils import PoseFeatureExtractor
        model = keras.models.load_model(MODELS_DIR / "model1_mlp.keras")
        scaler = joblib.load(MODELS_DIR / "model1_scaler.joblib")
        extractor = PoseFeatureExtractor(static_image_mode=False)
    else:
        model = keras.models.load_model(MODELS_DIR / "model2_cnn.keras")
    print(f"Using {which}. Press q to quit.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")

    history = deque(maxlen=SMOOTH_WINDOW)
    time_in_class = {c: 0.0 for c in CLASS_NAMES}
    bad_since, last_alert, alerts = None, 0.0, 0
    prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if MIRROR:
            frame = cv2.flip(frame, 1)
        now = time.time()
        dt, prev = now - prev, now

        probs = None
        if which == "model1":
            feats, result = extractor.extract(frame)
            extractor.draw(frame, result)
            if feats is not None:
                probs = model(scaler.transform(feats.reshape(1, -1)), training=False).numpy()[0]
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype("float32")[None]
            probs = model(x, training=False).numpy()[0]

        if probs is None:                                   # nobody in view
            history.clear()
            bad_since = None
            cv2.putText(frame, "No person detected", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        else:
            history.append(probs)
            avg = np.mean(history, axis=0)                  # smoothing over last frames
            label = CLASS_NAMES[int(avg.argmax())]
            score = int(round(100 * avg[CLASS_NAMES.index(GOOD_CLASS)]))
            time_in_class[label] += dt

            if label == GOOD_CLASS:
                bad_since = None
            else:
                bad_since = bad_since or now
                if now - bad_since >= args.alert_seconds and now - last_alert >= ALERT_COOLDOWN:
                    notify(f"Bad posture ({label}). Please sit upright.")
                    last_alert, alerts = now, alerts + 1

            color = (0, 200, 0) if label == GOOD_CLASS else (0, 0, 255)
            cv2.putText(frame, f"Posture: {label} ({avg.max() * 100:.0f}%)", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Ergonomic score: {score}/100", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if bad_since and now - bad_since >= args.alert_seconds:
                cv2.rectangle(frame, (0, frame.shape[0] - 50), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                cv2.putText(frame, "BAD POSTURE - SIT UPRIGHT", (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Posture & Ergonomics Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if extractor:
        extractor.close()

    total = sum(time_in_class.values())
    if total > 0:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rep = pd.DataFrame({"posture": CLASS_NAMES,
                            "seconds": [round(time_in_class[c], 1) for c in CLASS_NAMES]})
        rep["percent"] = (100 * rep["seconds"] / total).round(1)
        rep.to_csv(RESULTS_DIR / f"session_report_{stamp}.csv", index=False)
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.bar(rep["posture"], rep["seconds"],
               color=["green" if p == GOOD_CLASS else "crimson" for p in rep["posture"]])
        ax.set_ylabel("seconds"); ax.set_title("Session posture summary")
        fig.tight_layout(); fig.savefig(RESULTS_DIR / f"session_report_{stamp}.png", dpi=150)
        print("\nSession summary:\n", rep.to_string(index=False))
        print(f"Alerts shown: {alerts}\nReport saved in results/")


if __name__ == "__main__":
    main()

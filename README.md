# Posture & Ergonomics Monitoring using Deep Learning

**Course:** CS1218 – Deep Learning  
**Submitted to:** Dr. Akshat Agarwal  
**Team:** Rohit Soni (2024BTECH013) · Lucky Pandey (2024BTECH066) · Ankit Joshi (2024BTECH078)

---

## 1. Overview

People who study or work at a desk for hours often sit with a bent back, a forward head or
rounded shoulders. This causes neck and back pain and other musculoskeletal problems, and most
people do not notice bad posture while it is happening.

This project uses an ordinary **webcam** to detect a person's **sitting posture in real time**
and warns them when the posture stays unhealthy. Two deep learning models are trained on our own
dataset and compared, and the better one is used in the live monitoring system.

## 2. Objectives

1. Build a dataset of sitting-posture images from several people.
2. Train and compare two deep learning models for posture classification.
3. Evaluate both models fairly on people that were not used for training.
4. Build a real-time monitor with an ergonomic score, alerts and a session report.

## 3. Posture classes

| Class | What it looks like |
|---|---|
| `good` | Back straight, head over the shoulders, shoulders level |
| `forward_head` | Neck pushed forward toward the screen, back mostly straight |
| `slouching` | Back rounded, head and shoulders dropped and hunched |
| `leaning` | Body or head tilted to one side (left or right) |

## 4. System workflow

```
Webcam (OpenCV)
   -> Frame preprocessing (mirror, resize, normalise)
   -> MediaPipe Pose: body keypoints (nose, ears, shoulders)
   -> Feature creation: angles and ratios
   -> Model 1 (MLP on features)   |   Model 2 (MobileNetV2 CNN on the image)
   -> Evaluate and compare (accuracy, precision, recall, F1, confusion matrix, FPS)
   -> Best model goes into the live app
   -> Smoothing (last 10 frames) -> ergonomic score -> alert if bad posture lasts 10 s
   -> Session report (CSV + graph)
```

A copy of the flowchart is in `docs/flowchart.png`.

## 5. The two models

| | **Model 1 – Keypoint MLP** | **Model 2 – Image CNN** |
|---|---|---|
| Input | 8 pose features (angles and ratios) | Whole webcam image, 224 × 224 |
| Network | Multi-Layer Perceptron | MobileNetV2 with transfer learning |
| Strength | Very fast and light | Looks at the full image, not only the skeleton |
| Weakness | Depends on pose detection working | Affected by background, clothes, lighting |

### Model 1 – MLP (`train_model1.py`)
- Architecture: Input(8) → Dense 64 (ReLU) → Dropout 0.3 → Dense 32 (ReLU) → Dropout 0.2 → Dense 4 (Softmax)
- Loss: sparse categorical cross-entropy; optimizer: Adam (learning rate 0.001)
- Up to 150 epochs, batch size 32, early stopping on validation loss (patience 15)
- Class weights so that smaller classes are not ignored; features standardised with `StandardScaler`

### Model 2 – MobileNetV2 CNN (`train_model2.py`)
- MobileNetV2 pretrained on ImageNet, with a new head: Global Average Pooling → Dropout 0.3 → Dense 4 (Softmax)
- Augmentation (training only): random brightness, contrast, rotation, zoom
- **Phase 1:** base frozen, only the new top layers are trained (learning rate 0.001)
- **Phase 2 (fine-tuning):** last 30 layers of the base unfrozen, learning rate 0.00001, BatchNorm layers kept frozen
- Early stopping and class weights in both phases

### What is deep learning and what is not
- **Deep learning:** Model 1 (MLP), Model 2 (CNN), and MediaPipe Pose (a pretrained network we only *use* as a feature extractor, we do not train it).
- **Not deep learning:** angle and ratio calculation, scaling, smoothing over frames, the bad-posture timer and the alert logic.

## 6. Pose features (`src/pose_utils.py`)

Ratios are divided by the shoulder width, so they do not depend on how far the user is from the camera.

| Feature | Meaning |
|---|---|
| `shoulder_tilt` | Angle of the shoulder line against horizontal (leaning) |
| `head_tilt` | Angle of the line between the ears |
| `neck_angle` | Angle of the shoulder-midpoint → ear-midpoint line against vertical |
| `neck_vert_ratio` | Height of the ears above the shoulders ÷ shoulder width |
| `nose_vert_ratio` | Height of the nose above the shoulders ÷ shoulder width (drops when slouching) |
| `head_offset_x` | Sideways offset of the nose from the shoulder midpoint ÷ shoulder width |
| `head_depth` | Depth of the ears relative to the shoulders ÷ shoulder width (forward head) |
| `shoulder_width` | Shoulder width in the image (grows when the user leans toward the camera) |

## 7. Project structure

```
posture-ergonomics-dl/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py            all settings (classes, paths, alert time, ...)
│   ├── collect_data.py      record images from the webcam
│   ├── make_splits.py       train / val / test split by person
│   ├── extract_features.py  MediaPipe -> data/features.csv
│   ├── train_model1.py      train the MLP
│   ├── train_model2.py      train the MobileNetV2 CNN
│   ├── evaluate.py          compare the models on unseen test images
│   ├── live_app.py          real-time monitor with alerts and report
│   ├── pose_utils.py        landmarks -> features
│   ├── split_utils.py       dataset listing and person-wise split
│   └── plot_utils.py        training curves and confusion matrices
├── tests/test_logic.py      quick checks (no webcam / TensorFlow needed)
├── data/raw/<class>/        collected images
├── models/                  saved models
├── results/                 graphs, tables, session reports
└── docs/                    project brief and flowchart
```

## 8. Setup

Use **Python 3.10 or 3.11** (newer versions may not work with MediaPipe and TensorFlow).

```
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python tests/test_logic.py       # should print: All logic tests passed.
```

`mediapipe` is pinned to **0.10.14** on purpose: newer releases removed the `solutions` API this code uses.
Run every command from the project root folder.

## 9. Collecting the data

**Camera setup (keep it the same for every person and for the live demo)**
- Sit on a chair in front of the laptop, camera at **face level**, about an **arm's length (50–70 cm)** away.
- Head, **both shoulders** and some of the chest must be visible. If a shoulder leaves the frame, pose detection fails and the image is skipped.
- Light should fall on your face, not come from behind you.

**Recording**

```
python src/collect_data.py --person rohit
```

| Key | Action |
|---|---|
| 1 | record `good` |
| 2 | record `forward_head` |
| 3 | record `slouching` |
| 4 | record `leaning` |
| SPACE | stop recording |
| q | quit |

After a key press there is a 3-second countdown, then an image is saved every 0.5 s to
`data/raw/<class>/<person>__<timestamp>.jpg`. Use the same `--person` name every time for the same person.

**Guidelines**
- Record each posture for about 2–3 minutes, moving slightly so the images are not identical. For `leaning`, record both left and right.
- Make the postures clearly different from each other; do not mix them (no head pushed forward while recording `good`).
- Aim for **500–1000+ images per class** from **at least 6–8 different people**, with different clothes, lighting and slight camera changes.
- If you record on different laptops, copy all `data/raw/<class>/` folders into one laptop; file names are unique, so nothing is overwritten.
- Delete bad images (shoulders cut off, face covered).

## 10. Running the pipeline

```
python src/make_splits.py        # 1. split by person into train / val / test
python src/extract_features.py   # 2. MediaPipe features -> data/features.csv
python src/train_model1.py       # 3. train Model 1
python src/train_model2.py       # 4. train Model 2 (downloads ImageNet weights once, needs internet)
python src/evaluate.py           # 5. compare both models
python src/live_app.py           # 6. real-time monitoring (press q to quit)
```

Useful options:
- `python src/live_app.py --model model1` (or `model2`) to force a model; by default the winner from `evaluate.py` is used.
- `python src/live_app.py --alert_seconds 5` to change how long bad posture must last before an alert.
- `python src/make_splits.py --random_split` only for a quick code test when fewer than 3 people are available. Its accuracy is **not trustworthy** (the same person is in train and test).

## 11. How the data is split

The images are split **by person** (about 70 % / 15 % / 15 % of the people for train / validation / test).
No person appears in more than one split. This matters because if photos of the same person were in both
train and test, the model could recognise the person and the accuracy would look better than it really is.
At least 3 people are required; 6–8 or more are recommended.

## 12. Evaluation

Both models are tested on the **same unseen test images** and compared with:
- accuracy, macro precision, macro recall, macro F1-score
- confusion matrix and per-class report
- speed in milliseconds per frame and FPS, measured for the full per-frame pipeline

The model with the higher macro F1 is saved in `models/best_model.json` and used by the live app.
Training curves (loss and accuracy) are saved to check for overfitting.

**Outputs** (created after running the pipeline)
- `models/`: `model1_mlp.keras`, `model1_scaler.joblib`, `model2_cnn.keras`, `best_model.json`
- `results/`: `model1_training.png`, `model2_training.png`, `comparison.csv`, `confusion_model1.png`, `confusion_model2.png`, `report_model1.txt`, `report_model2.txt`, session reports

### Results (fill in after running `evaluate.py`)

| Model | Accuracy | Precision | Recall | F1 | FPS |
|---|---|---|---|---|---|
| Model 1 – Keypoint MLP | | | | | |
| Model 2 – MobileNetV2 CNN | | | | | |

Only report numbers that `evaluate.py` actually produces on your own data.

## 13. Live monitoring

1. The webcam frame is mirrored and passed to the selected model.
2. For Model 1, MediaPipe finds the keypoints and the features are computed; for Model 2 the frame is resized to 224 × 224.
3. The model predicts the posture class. Predictions are **averaged over the last 10 frames** to avoid flicker.
4. The **ergonomic score (0–100)** is the smoothed probability of `good`.
5. If bad posture lasts **10 seconds**, a red banner appears and a desktop notification is sent (60 s cooldown between alerts).
6. If nobody is detected, the timers pause.
7. When you press **q**, a session report (`results/session_report_<time>.csv` and `.png`) shows how many seconds were spent in each posture.

## 14. Settings you can change (`src/config.py`)

`CLASS_NAMES`, `CAMERA_INDEX`, `MIRROR`, `SMOOTH_WINDOW`, `ALERT_SECONDS`, `ALERT_COOLDOWN`, `IMG_SIZE`, `VAL_FRACTION`, `TEST_FRACTION`.
`MIRROR` must be the same when collecting data and when running the live app.

## 15. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'config'` | Run scripts as `python src/<script>.py` from the project root |
| `No module named pandas / tensorflow` | Activate the virtual environment, then `pip install -r requirements.txt` |
| MediaPipe has no `solutions` | `pip install mediapipe==0.10.14` |
| Webcam does not open | Close Zoom/Teams/Camera app, or use `--camera 1` |
| Many images "skipped" in `extract_features.py` | Both shoulders and the nose must be visible; improve framing and lighting |
| "Need at least 3 people" | Collect data from more people |
| Model 2 training is slow | Use Google Colab with a free GPU, or reduce the number of images |
| One class is always confused with another | Make the postures more different and record more varied data |

## 16. Limitations

- Works best with the camera in front of the user, the upper body visible and reasonable lighting.
- A single webcam gives no true 3D depth, so forward-head detection is approximate.
- Accuracy depends on the quality and variety of the collected data (people, clothes, lighting, camera angle).
- The system is a posture-awareness aid, not a medical diagnosis tool.

## 17. Possible future work

- A rule-based ergonomic angle score (RULA-style) shown next to the model's score.
- More classes (for example leaning back) and more participants.
- A temporal model (LSTM) that uses a sequence of frames instead of a single frame.
- Exporting the model to a lighter format for low-end laptops.

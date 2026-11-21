"""Step 3: run MediaPipe on every image and save the pose features to data/features.csv.
This CSV is the training data for Model 1 (the keypoint MLP)."""
import cv2
import pandas as pd
from tqdm import tqdm

from config import ROOT, FEATURES_CSV, FEATURE_NAMES
from pose_utils import PoseFeatureExtractor
from split_utils import load_splits


def main():
    df = load_splits()
    extractor = PoseFeatureExtractor(static_image_mode=True)
    rows, skipped = [], {}

    for r in tqdm(df.itertuples(index=False), total=len(df), desc="Extracting"):
        img = cv2.imread(str(ROOT / r.filepath))
        feats = None
        if img is not None:
            feats, _ = extractor.extract(img)
        if feats is None:
            skipped[r.label] = skipped.get(r.label, 0) + 1
            continue
        rows.append({"filepath": r.filepath, "label": r.label, "person": r.person,
                     "split": r.split, **dict(zip(FEATURE_NAMES, map(float, feats)))})
    extractor.close()

    out = pd.DataFrame(rows)
    out.to_csv(FEATURES_CSV, index=False)
    print(f"\nSaved {len(out)} rows to {FEATURES_CSV}")
    if skipped:
        print("Skipped (no clear pose found):", skipped)
    print(out.groupby(["split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()

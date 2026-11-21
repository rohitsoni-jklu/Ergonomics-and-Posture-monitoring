"""Dataset listing and person-wise train/val/test splitting.

Why person-wise? If photos of the same person are in both train and test, the model
can "recognise the person" and the accuracy looks better than it really is.
Images must be named  <person>__<timestamp>.jpg  (collect_data.py does this).
"""
import random
import pandas as pd

from config import RAW_DIR, ROOT, CLASS_NAMES, SEED, VAL_FRACTION, TEST_FRACTION, SPLITS_CSV

IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_dataset():
    rows = []
    for label in CLASS_NAMES:
        folder = RAW_DIR / label
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                person = p.stem.split("__")[0].lower()
                rows.append({"filepath": p.relative_to(ROOT).as_posix(),
                             "label": label, "person": person})
    return pd.DataFrame(rows)


def make_splits(allow_random=False):
    df = list_dataset()
    if df.empty:
        raise SystemExit("No images found in data/raw/<class>/. Run collect_data.py first.")
    missing = [c for c in CLASS_NAMES if c not in set(df["label"])]
    if missing:
        raise SystemExit(f"No images for classes: {missing}")

    persons = sorted(df["person"].unique())
    rng = random.Random(SEED)

    if len(persons) >= 3:
        rng.shuffle(persons)
        n = len(persons)
        n_test = max(1, round(n * TEST_FRACTION))
        n_val = max(1, round(n * VAL_FRACTION))
        test_p = set(persons[:n_test])
        val_p = set(persons[n_test:n_test + n_val])
        df["split"] = df["person"].map(
            lambda p: "test" if p in test_p else ("val" if p in val_p else "train"))
        if n < 6:
            print(f"WARNING: only {n} people. Person-wise splits will be small; "
                  "collect data from 6+ people for reliable results.")
    elif allow_random:
        print("WARNING: fewer than 3 people -> using a RANDOM split. Test accuracy will be "
              "optimistic because the same people appear in train and test.")
        df["split"] = "train"
        for _, g in df.groupby("label"):
            idx = list(g.index)
            rng.shuffle(idx)
            n = len(idx)
            df.loc[idx[: int(n * TEST_FRACTION)], "split"] = "test"
            df.loc[idx[int(n * TEST_FRACTION): int(n * (TEST_FRACTION + VAL_FRACTION))], "split"] = "val"
    else:
        raise SystemExit(
            f"Found {len(persons)} person(s): {persons}. Need at least 3 for a person-wise split.\n"
            "Collect data from more people, or run with --random_split (not recommended).")

    df.to_csv(SPLITS_CSV, index=False)
    return df


def load_splits():
    if not SPLITS_CSV.exists():
        raise SystemExit("data/splits.csv not found. Run:  python src/make_splits.py")
    return pd.read_csv(SPLITS_CSV)

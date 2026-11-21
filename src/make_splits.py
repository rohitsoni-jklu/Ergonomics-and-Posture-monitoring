"""Step 2: split the collected images into train / val / test (by person)."""
import argparse
from split_utils import make_splits
from config import SPLITS_CSV

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--random_split", action="store_true",
                    help="allow a random split when fewer than 3 people (not recommended)")
    args = ap.parse_args()
    df = make_splits(allow_random=args.random_split)
    print(f"Saved {SPLITS_CSV}\n")
    print("People per split:")
    print(df.groupby("split")["person"].unique().to_string(), "\n")
    print("Images per class and split:")
    print(df.pivot_table(index="label", columns="split", values="filepath", aggfunc="count").fillna(0).astype(int))

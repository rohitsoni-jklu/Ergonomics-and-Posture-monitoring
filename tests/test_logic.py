"""Quick checks that do not need TensorFlow, MediaPipe or a webcam.
Run:  python tests/test_logic.py"""
import sys
from pathlib import Path
from types import SimpleNamespace as P

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from config import FEATURE_NAMES
from pose_utils import compute_features


def lms(nose, lear, rear, ls, rs, vis=1.0):
    lm = [P(x=0.5, y=0.5, z=0.0, visibility=0.0) for _ in range(33)]
    for idx, (x, y, z) in {0: nose, 7: lear, 8: rear, 11: ls, 12: rs}.items():
        lm[idx] = P(x=x, y=y, z=z, visibility=vis)
    return lm


def test_good_vs_slouch():
    good = compute_features(lms((0.5, 0.30, 0), (0.55, 0.32, 0), (0.45, 0.32, 0),
                                (0.62, 0.55, 0), (0.38, 0.55, 0)))
    slouch = compute_features(lms((0.5, 0.45, -0.1), (0.55, 0.46, -0.1), (0.45, 0.46, -0.1),
                                  (0.62, 0.55, 0), (0.38, 0.55, 0)))
    assert len(good) == len(FEATURE_NAMES)
    i = FEATURE_NAMES.index("nose_vert_ratio")
    assert good[i] > slouch[i], "head should be lower relative to shoulders when slouching"
    assert abs(good[FEATURE_NAMES.index("shoulder_tilt")]) < 1e-3


def test_lean():
    lean = compute_features(lms((0.35, 0.30, 0), (0.40, 0.32, 0), (0.30, 0.32, 0),
                                (0.62, 0.55, 0), (0.38, 0.55, 0)))
    assert lean[FEATURE_NAMES.index("head_offset_x")] < -0.3


def test_low_visibility_returns_none():
    assert compute_features(lms((0.5, 0.3, 0), (0.55, 0.3, 0), (0.45, 0.3, 0),
                                (0.6, 0.5, 0), (0.4, 0.5, 0), vis=0.1)) is None


if __name__ == "__main__":
    test_good_vs_slouch(); test_lean(); test_low_visibility_returns_none()
    print("All logic tests passed.")

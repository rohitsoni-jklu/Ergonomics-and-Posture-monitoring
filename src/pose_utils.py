"""Pose landmarks -> ergonomic features (angles and ratios).

compute_features() only needs objects with .x .y .z .visibility, so it can be tested
without MediaPipe. PoseFeatureExtractor wraps MediaPipe Pose.
"""
import math
import numpy as np
import cv2

from config import FEATURE_NAMES

# MediaPipe Pose landmark indices
NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 0, 7, 8, 11, 12
MIN_VIS = 0.5
EPS = 1e-6


def _mid(a, b):
    return ((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)


def _tilt_deg(p, q):
    """Angle of the line p-q against the horizontal (points ordered left->right in the image)."""
    if p.x > q.x:
        p, q = q, p
    return math.degrees(math.atan2(q.y - p.y, (q.x - p.x) + EPS))


def compute_features(lm):
    """lm: list of 33 landmarks. Returns a numpy array (len(FEATURE_NAMES)) or None."""
    nose, ls, rs = lm[NOSE], lm[L_SHOULDER], lm[R_SHOULDER]
    if min(nose.visibility, ls.visibility, rs.visibility) < MIN_VIS:
        return None

    sh_x, sh_y, sh_z = _mid(ls, rs)
    sw = math.hypot(ls.x - rs.x, ls.y - rs.y) + EPS          # shoulder width (scale reference)

    ears = [e for e in (lm[L_EAR], lm[R_EAR]) if e.visibility >= MIN_VIS]
    if len(ears) == 2:
        ex, ey, ez = _mid(ears[0], ears[1])
        head_tilt = _tilt_deg(ears[0], ears[1])
    elif len(ears) == 1:
        ex, ey, ez = ears[0].x, ears[0].y, ears[0].z
        head_tilt = 0.0
    else:                                                    # no ear visible: use the nose
        ex, ey, ez = nose.x, nose.y, nose.z
        head_tilt = 0.0

    shoulder_tilt = _tilt_deg(ls, rs)
    neck_angle = math.degrees(math.atan2(abs(ex - sh_x), (sh_y - ey) + EPS))
    neck_vert_ratio = (sh_y - ey) / sw
    nose_vert_ratio = (sh_y - nose.y) / sw
    head_offset_x = (nose.x - sh_x) / sw
    head_depth = (ez - sh_z) / sw

    feats = [shoulder_tilt, head_tilt, neck_angle, neck_vert_ratio,
             nose_vert_ratio, head_offset_x, head_depth, sw]
    assert len(feats) == len(FEATURE_NAMES)
    return np.array(feats, dtype=np.float32)


class PoseFeatureExtractor:
    def __init__(self, static_image_mode=False):
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Your MediaPipe version has no 'solutions' API. "
                "Run:  pip install mediapipe==0.10.14")
        self._mp = mp
        self._drawing = mp.solutions.drawing_utils
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def extract(self, bgr_image):
        """Returns (features or None, mediapipe_result)."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None, result
        return compute_features(result.pose_landmarks.landmark), result

    def draw(self, bgr_image, result):
        if result is not None and result.pose_landmarks:
            self._drawing.draw_landmarks(
                bgr_image, result.pose_landmarks, self._mp.solutions.pose.POSE_CONNECTIONS)

    def close(self):
        self._pose.close()

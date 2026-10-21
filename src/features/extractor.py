"""MediaPipe Hands + Pose landmark extraction.

Uses the Tasks-vision API (HandLandmarker, PoseLandmarker) — the same API
family and the same model files the browser uses for live inference
(web/app.js), not the legacy `mp.solutions` API this originally used.

That switch matters: a diagnostic comparing both APIs on the same 222
dataset images (both detected a hand) found the already-trained classifier
scored 100% on legacy-API landmarks but only 88.3% on Tasks-API landmarks —
every error one-directional (legacy right, Tasks wrong; never the reverse).
Same 21-point format, but a different underlying landmark-detection model
with meaningfully different coordinate precision. Training on legacy-API
landmarks while the browser infers with Tasks-API landmarks is a train/serve
skew bug, not generic webcam distribution shift — it meant the classifier
was never actually trained on the kind of landmarks it sees live. Rebuilding
the landmark cache with this extractor and retraining on it removes that
specific bug (real webcam/lighting/hand distribution shift is separate and
still expected to some degree).

Returns a fixed-size feature vector per image so downstream models get
consistent input shapes regardless of how many hands/pose were detected.
"""
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

HAND_LANDMARKS = 21  # MediaPipe Hands: 21 points per hand
POSE_LANDMARKS = 33  # MediaPipe Pose: 33 body points
COORDS = 3  # x, y, z per landmark

# Final feature vector: [left_hand(21*3) + right_hand(21*3) + pose(33*3) + presence(3)]
# presence flags: [left_hand_present, right_hand_present, pose_present]
FEATURE_DIM = HAND_LANDMARKS * COORDS * 2 + POSE_LANDMARKS * COORDS + 3

# Same model files web/app.js loads from this CDN at runtime — using the
# identical asset is what makes this match live browser inference.
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_CACHE_DIR = Path("data/mediapipe_models")


def _ensure_model(url: str) -> str:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_CACHE_DIR / url.rsplit("/", 1)[-1]
    if not path.exists():
        urlretrieve(url, path)
    return str(path)


@dataclass
class LandmarkResult:
    features: np.ndarray  # shape (FEATURE_DIM,)
    left_hand_present: bool
    right_hand_present: bool
    pose_present: bool


class LandmarkExtractor:
    """Wraps Tasks-vision HandLandmarker + PoseLandmarker into a single
    extractor, run in IMAGE mode (these are static dataset photos, not
    video). Use as a context manager or call .close() when done.
    """

    def __init__(self, static_mode: bool = True, min_detection_confidence: float = 0.5):
        del static_mode  # IMAGE running mode is always single-shot; kept for call-site compat
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_ensure_model(HAND_MODEL_URL)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=min_detection_confidence,
            )
        )
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_ensure_model(POSE_MODEL_URL)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
            )
        )

    def extract(self, image) -> LandmarkResult:
        """Extract landmarks from a PIL Image or numpy array (RGB)."""
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        hand_result = self.hands.detect(mp_image)
        pose_result = self.pose.detect(mp_image)

        left_hand = np.zeros(HAND_LANDMARKS * COORDS, dtype=np.float32)
        right_hand = np.zeros(HAND_LANDMARKS * COORDS, dtype=np.float32)
        pose = np.zeros(POSE_LANDMARKS * COORDS, dtype=np.float32)

        left_present = right_present = pose_present = False

        if hand_result.hand_landmarks and hand_result.handedness:
            for lm_list, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = handedness[0].category_name  # "Left" or "Right"
                flat = np.array(
                    [[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32
                ).flatten()
                if label == "Left":
                    left_hand, left_present = flat, True
                else:
                    right_hand, right_present = flat, True

        if pose_result.pose_landmarks:
            pose = np.array(
                [[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]],
                dtype=np.float32,
            ).flatten()
            pose_present = True

        presence = np.array(
            [float(left_present), float(right_present), float(pose_present)],
            dtype=np.float32,
        )
        features = np.concatenate([left_hand, right_hand, pose, presence])

        return LandmarkResult(
            features=features,
            left_hand_present=left_present,
            right_hand_present=right_present,
            pose_present=pose_present,
        )

    def close(self):
        self.hands.close()
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

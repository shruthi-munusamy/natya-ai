"""MediaPipe Hands + Pose landmark extraction.

Returns a fixed-size feature vector per image so downstream models
get consistent input shapes regardless of how many hands/pose were detected.
"""
from dataclasses import dataclass
import numpy as np
import mediapipe as mp
from PIL import Image

HAND_LANDMARKS = 21  # MediaPipe Hands: 21 points per hand
POSE_LANDMARKS = 33  # MediaPipe Pose: 33 body points
COORDS = 3  # x, y, z per landmark

# Final feature vector: [left_hand(21*3) + right_hand(21*3) + pose(33*3) + presence(3)]
# presence flags: [left_hand_present, right_hand_present, pose_present]
FEATURE_DIM = HAND_LANDMARKS * COORDS * 2 + POSE_LANDMARKS * COORDS + 3


@dataclass
class LandmarkResult:
    features: np.ndarray  # shape (FEATURE_DIM,)
    left_hand_present: bool
    right_hand_present: bool
    pose_present: bool


class LandmarkExtractor:
    """Wraps MediaPipe Hands + Pose into a single extractor.

    Use as a context manager or call .close() when done — MediaPipe
    holds graph resources that need explicit cleanup.
    """

    def __init__(self, static_mode: bool = True, min_detection_confidence: float = 0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_mode,
            min_detection_confidence=min_detection_confidence,
        )

    def extract(self, image) -> LandmarkResult:
        """Extract landmarks from a PIL Image or numpy array (RGB)."""
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        hand_results = self.hands.process(image)
        pose_results = self.pose.process(image)

        left_hand = np.zeros(HAND_LANDMARKS * COORDS, dtype=np.float32)
        right_hand = np.zeros(HAND_LANDMARKS * COORDS, dtype=np.float32)
        pose = np.zeros(POSE_LANDMARKS * COORDS, dtype=np.float32)

        left_present = right_present = pose_present = False

        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_lms, handedness in zip(
                hand_results.multi_hand_landmarks, hand_results.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"
                flat = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                    dtype=np.float32,
                ).flatten()
                if label == "Left":
                    left_hand, left_present = flat, True
                else:
                    right_hand, right_present = flat, True

        if pose_results.pose_landmarks:
            pose = np.array(
                [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark],
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

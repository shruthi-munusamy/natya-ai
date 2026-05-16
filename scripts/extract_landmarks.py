"""Pre-extract landmarks for the whole Samarth dataset and cache as a .npz file.

Run once. Downstream training reads the cache, not the images.
"""
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.loader import load_mudra_dataset
from src.features.extractor import LandmarkExtractor, FEATURE_DIM


def main():
    ds = load_mudra_dataset()
    train = ds["train"]
    n = len(train)

    features = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    label_names = np.empty(n, dtype=object)
    gesture_types = np.empty(n, dtype=object)
    hand_present = np.zeros((n, 2), dtype=bool)  # [left, right]
    pose_present = np.zeros(n, dtype=bool)

    with LandmarkExtractor(static_mode=True) as extractor:
        for i, sample in enumerate(tqdm(train, desc="Extracting landmarks")):
            result = extractor.extract(sample["image"])
            features[i] = result.features
            labels[i] = sample["label_id"]
            label_names[i] = sample["label"]
            gesture_types[i] = sample["gesture_type"]
            hand_present[i] = [result.left_hand_present, result.right_hand_present]
            pose_present[i] = result.pose_present

    out_path = Path("data/landmarks.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=features,
        labels=labels,
        label_names=label_names,
        gesture_types=gesture_types,
        hand_present=hand_present,
        pose_present=pose_present,
    )

    print(f"\nSaved {n} samples to {out_path}")
    print(f"Feature dim: {FEATURE_DIM}")
    print(f"Left hand detected: {hand_present[:, 0].sum()}/{n} "
          f"({100 * hand_present[:, 0].mean():.1f}%)")
    print(f"Right hand detected: {hand_present[:, 1].sum()}/{n} "
          f"({100 * hand_present[:, 1].mean():.1f}%)")
    print(f"Pose detected: {pose_present.sum()}/{n} "
          f"({100 * pose_present.mean():.1f}%)")
    print(f"At least one hand: {hand_present.any(axis=1).sum()}/{n} "
          f"({100 * hand_present.any(axis=1).mean():.1f}%)")


if __name__ == "__main__":
    main()

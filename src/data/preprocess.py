"""Preprocessing for asamyukta classifier training."""
import numpy as np
from sklearn.model_selection import train_test_split

HAND_LANDMARKS = 21
COORDS = 3
HAND_FEATURE_DIM = HAND_LANDMARKS * COORDS  # 63


def load_landmark_cache(path: str = "data/landmarks.npz"):
    return np.load(path, allow_pickle=True)


def filter_asamyukta(gesture_types: np.ndarray, hand_present: np.ndarray):
    """Mask: single-hand samples with at least one hand detected."""
    is_single = np.array([
        "single" in str(g).lower() for g in gesture_types
    ])
    has_hand = hand_present.any(axis=1)
    return is_single & has_hand


def extract_primary_hand(features: np.ndarray, hand_present: np.ndarray) -> np.ndarray:
    """One 63-dim hand vector per sample. Left if present, else right.
    Horizontal-flip augmentation at training time makes the model hand-agnostic.
    """
    left = features[:, :HAND_FEATURE_DIM]
    right = features[:, HAND_FEATURE_DIM:2 * HAND_FEATURE_DIM]
    return np.where(hand_present[:, 0:1], left, right)


def normalize_landmarks(hand_features: np.ndarray) -> np.ndarray:
    """Wrist-centered, scale-normalized hand coords.

    Translation invariance: subtract wrist (point 0) from all 21 keypoints.
    Scale invariance: divide by distance from wrist to middle-finger MCP (point 9).
    """
    n = len(hand_features)
    points = hand_features.reshape(n, HAND_LANDMARKS, COORDS)
    wrist = points[:, 0:1, :]
    centered = points - wrist
    scale = np.linalg.norm(centered[:, 9, :], axis=-1, keepdims=True).reshape(n, 1, 1)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (centered / scale).reshape(n, HAND_FEATURE_DIM)


def horizontal_flip(hand_features: np.ndarray) -> np.ndarray:
    """Flip x-coordinates. After wrist-centering, x is centered around 0,
    so negating it produces a mirrored hand."""
    n = len(hand_features)
    points = hand_features.reshape(n, HAND_LANDMARKS, COORDS).copy()
    points[:, :, 0] = -points[:, :, 0]
    return points.reshape(n, HAND_FEATURE_DIM)


def random_rotate(
    hand_features: np.ndarray,
    max_roll_deg: float = 180.0,
    max_tilt_deg: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random 3D rotation about the wrist (the origin after normalization).

    The dataset photographs hands almost entirely side-on and at fixed
    tilts per class — e.g. Chandrakala's wrist->MCP axis sits at 48-79 deg
    from vertical in 80% of samples, so the model learned "index finger
    horizontal" as part of Chandrakala and called an upright one Suchi live.
    Mudra identity doesn't depend on in-plane orientation (pose context
    carries that signal separately), so full roll is safe.

    Tilt (pitch/yaw) defaults to off. Training with +/-35 deg tilt made live
    Pataka read as Sarpasirsha: the two differ mainly by a slight cup and
    the hand angled back, so tilting flat Pataka hands manufactured that
    Sarpasirsha cue (held-out Pataka->Sarpasirsha 0% -> 5%; roll-only: 1%,
    with the Chandrakala/Bramaram fixes intact).
    """
    rng = rng or np.random.default_rng()
    n = len(hand_features)
    points = hand_features.reshape(n, HAND_LANDMARKS, COORDS)
    roll = np.radians(rng.uniform(-max_roll_deg, max_roll_deg, n))
    pitch = np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg, n))
    yaw = np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg, n))

    def rot(axis_a, axis_b, theta):
        r = np.tile(np.eye(3), (n, 1, 1))
        c, s = np.cos(theta), np.sin(theta)
        r[:, axis_a, axis_a] = c
        r[:, axis_a, axis_b] = -s
        r[:, axis_b, axis_a] = s
        r[:, axis_b, axis_b] = c
        return r

    r = rot(0, 1, roll) @ rot(1, 2, pitch) @ rot(0, 2, yaw)
    rotated = points @ r.transpose(0, 2, 1)
    return rotated.reshape(n, HAND_FEATURE_DIM).astype(hand_features.dtype)


def build_asamyukta_dataset(
    cache_path: str = "data/landmarks.npz",
    val_size: float = 0.2,
    seed: int = 42,
):
    data = load_landmark_cache(cache_path)
    features = data["features"]
    labels = data["labels"]
    label_names = data["label_names"]
    gesture_types = data["gesture_types"]
    hand_present = data["hand_present"]

    print(f"Gesture types in cache: {np.unique(gesture_types)}")

    mask = filter_asamyukta(gesture_types, hand_present)
    features = features[mask]
    labels = labels[mask]
    label_names = label_names[mask]
    hand_present = hand_present[mask]
    print(f"Single-hand with hand detected: {len(features)}")

    primary = extract_primary_hand(features, hand_present)
    normalized = normalize_landmarks(primary)

    unique_ids = np.unique(labels)
    id_to_idx = {lid: i for i, lid in enumerate(unique_ids)}
    y = np.array([id_to_idx[l] for l in labels], dtype=np.int64)
    class_names = [str(label_names[labels == lid][0]) for lid in unique_ids]

    x_train, x_val, y_train, y_val = train_test_split(
        normalized, y, test_size=val_size, stratify=y, random_state=seed
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "class_names": class_names,
        "n_classes": len(unique_ids),
        "n_train": len(y_train),
        "n_val": len(y_val),
    }


TWO_HAND_FEATURE_DIM = 2 * HAND_LANDMARKS * COORDS  # 126


def filter_samyukta(gesture_types: np.ndarray, hand_present: np.ndarray):
    """Mask: double-hand samples with BOTH hands detected."""
    is_double = np.array([
        "double" in str(g).lower() for g in gesture_types
    ])
    both = hand_present[:, 0] & hand_present[:, 1]
    return is_double & both


def normalize_two_hands(features: np.ndarray) -> np.ndarray:
    """Two-hand normalization in shared coordinate space.

    Origin = midpoint of the two wrists.
    Scale = average wrist-to-MCP distance across both hands.

    This preserves inter-hand spatial relationships (which IS the samyukta
    signal — e.g., palms touching for Anjali) in the same units as each
    hand's internal geometry. Different from asamyukta normalization, which
    centers each hand on its own wrist.
    """
    n = len(features)
    left = features[:, :HAND_FEATURE_DIM].reshape(n, HAND_LANDMARKS, COORDS)
    right = features[:, HAND_FEATURE_DIM:2 * HAND_FEATURE_DIM].reshape(n, HAND_LANDMARKS, COORDS)

    wrist_l = left[:, 0:1, :]
    wrist_r = right[:, 0:1, :]
    centroid = (wrist_l + wrist_r) / 2

    left_c = left - centroid
    right_c = right - centroid

    scale_l = np.linalg.norm(left[:, 9, :] - left[:, 0, :], axis=-1)
    scale_r = np.linalg.norm(right[:, 9, :] - right[:, 0, :], axis=-1)
    scale = ((scale_l + scale_r) / 2).reshape(n, 1, 1)
    scale = np.where(scale > 1e-6, scale, 1.0)

    two_hands = np.concatenate([left_c / scale, right_c / scale], axis=1)
    return two_hands.reshape(n, TWO_HAND_FEATURE_DIM)


def horizontal_flip_two_hands(two_hand_features: np.ndarray) -> np.ndarray:
    """Flip x and swap hand identities. Most samyuktas are bilaterally
    symmetric (Anjali, Kapota, Karkata), so swap-augmentation is safe and
    teaches the model to be invariant to MediaPipe's L/R labeling noise."""
    n = len(two_hand_features)
    points = two_hand_features.reshape(n, 2, HAND_LANDMARKS, COORDS).copy()
    points[:, :, :, 0] = -points[:, :, :, 0]
    points = points[:, [1, 0], :, :]
    return points.reshape(n, TWO_HAND_FEATURE_DIM)


SAMYUKTA_MIN_PER_CLASS = 30


def build_samyukta_dataset(
    cache_path: str = "data/landmarks.npz",
    val_size: float = 0.2,
    seed: int = 42,
    min_per_class: int = SAMYUKTA_MIN_PER_CLASS,
):
    data = load_landmark_cache(cache_path)
    features = data["features"]
    labels = data["labels"]
    label_names = data["label_names"]
    gesture_types = data["gesture_types"]
    hand_present = data["hand_present"]

    mask = filter_samyukta(gesture_types, hand_present)
    features = features[mask]
    labels = labels[mask]
    label_names = label_names[mask]
    print(f"Double-hand with both hands detected: {len(features)}")

    unique_ids, counts = np.unique(labels, return_counts=True)
    keep_ids = set(unique_ids[counts >= min_per_class].tolist())
    dropped = [(str(label_names[labels == lid][0]), int(counts[i]))
               for i, lid in enumerate(unique_ids) if lid not in keep_ids]
    if dropped:
        print(f"Dropping {len(dropped)} classes with <{min_per_class} samples: "
              + ", ".join(f"{n}({c})" for n, c in dropped))
    keep_mask = np.isin(labels, list(keep_ids))
    features = features[keep_mask]
    labels = labels[keep_mask]
    label_names = label_names[keep_mask]
    print(f"After filtering: {len(features)} samples across {len(keep_ids)} classes")

    normalized = normalize_two_hands(features)

    unique_ids = np.unique(labels)
    id_to_idx = {lid: i for i, lid in enumerate(unique_ids)}
    y = np.array([id_to_idx[l] for l in labels], dtype=np.int64)
    class_names = [str(label_names[labels == lid][0]) for lid in unique_ids]

    x_train, x_val, y_train, y_val = train_test_split(
        normalized, y, test_size=val_size, stratify=y, random_state=seed
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "class_names": class_names,
        "n_classes": len(unique_ids),
        "n_train": len(y_train),
        "n_val": len(y_val),
    }

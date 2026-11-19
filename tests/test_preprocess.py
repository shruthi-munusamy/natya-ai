import numpy as np

from src.data.preprocess import (
    HAND_FEATURE_DIM,
    TWO_HAND_FEATURE_DIM,
    horizontal_flip,
    horizontal_flip_two_hands,
    normalize_landmarks,
    normalize_two_hands,
    random_rotate,
)

rng = np.random.default_rng(0)


def test_normalize_landmarks_centers_wrist_and_scales_to_mcp():
    hands = rng.normal(size=(4, HAND_FEATURE_DIM)).astype(np.float32)
    points = normalize_landmarks(hands).reshape(4, 21, 3)
    np.testing.assert_allclose(points[:, 0], 0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(points[:, 9], axis=-1), 1, atol=1e-5)


def test_horizontal_flip_is_an_involution():
    hands = normalize_landmarks(rng.normal(size=(4, HAND_FEATURE_DIM)))
    np.testing.assert_allclose(horizontal_flip(horizontal_flip(hands)), hands)


def test_random_rotate_preserves_distances_from_wrist():
    hands = normalize_landmarks(rng.normal(size=(4, HAND_FEATURE_DIM)).astype(np.float32))
    rotated = random_rotate(hands, max_tilt_deg=30, rng=np.random.default_rng(1))
    before = np.linalg.norm(hands.reshape(4, 21, 3), axis=-1)
    after = np.linalg.norm(rotated.reshape(4, 21, 3), axis=-1)
    np.testing.assert_allclose(after, before, atol=1e-5)
    assert rotated.dtype == hands.dtype


def test_two_hand_normalize_and_flip():
    features = rng.normal(size=(3, TWO_HAND_FEATURE_DIM))
    normalized = normalize_two_hands(features)
    assert normalized.shape == (3, TWO_HAND_FEATURE_DIM)
    np.testing.assert_allclose(
        horizontal_flip_two_hands(horizontal_flip_two_hands(normalized)), normalized
    )

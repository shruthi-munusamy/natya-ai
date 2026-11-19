import pytest
import torch

from src.models import classifier as c


@pytest.mark.parametrize("model_cls, dim", [
    (c.MLPClassifier, c.HAND_FEATURE_DIM),
    (c.AttentionClassifier, c.HAND_FEATURE_DIM),
    (c.TwoHandMLPClassifier, c.TWO_HAND_FEATURE_DIM),
    (c.TwoHandAttentionClassifier, c.TWO_HAND_FEATURE_DIM),
])
def test_forward_shape(model_cls, dim):
    model = model_cls(n_classes=5).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, dim))
    assert logits.shape == (2, 5)

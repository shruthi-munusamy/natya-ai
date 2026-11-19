"""Export trained classifiers to ONNX for onnxruntime-web, and check the
ONNX outputs match PyTorch so runtime drift is caught before deployment.
"""
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.models.classifier import (
    AttentionClassifier,
    MLPClassifier,
    TwoHandAttentionClassifier,
    TwoHandMLPClassifier,
    HAND_FEATURE_DIM,
    TWO_HAND_FEATURE_DIM,
)


MODEL_REGISTRY = {
    "mlp": MLPClassifier,
    "attention": AttentionClassifier,
    "mlp_two_hand": TwoHandMLPClassifier,
    "attention_two_hand": TwoHandAttentionClassifier,
}


def export(checkpoint_path: str, out_path: str, feature_dim: int, arch_key: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    model_cls = MODEL_REGISTRY[arch_key]
    model = model_cls(n_classes=len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.randn(1, feature_dim, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        # torch>=2.6 defaults to the dynamo exporter, whose graphs make
        # quantize_dynamic fail with a ShapeInferenceError. Must stay False.
        dynamo=False,
    )

    onnx.checker.check_model(onnx.load(out_path))

    with torch.no_grad():
        torch_out = model(dummy).numpy()
    session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"landmarks": dummy.numpy()})[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"  {out_path}")
    print(f"  max abs diff (PyTorch vs ONNX): {max_diff:.2e}")
    if max_diff > 1e-4:
        raise RuntimeError(f"ONNX output drift too large: {max_diff}")
    print(f"  classes: {len(class_names)}")
    print("  ok\n")
    return class_names


def main():
    Path("checkpoints").mkdir(exist_ok=True)
    Path("web/models").mkdir(parents=True, exist_ok=True)

    print("Exporting asamyukta (attention)...")
    asamyukta_classes = export(
        "checkpoints/asamyukta_attention.pt",
        "web/models/asamyukta.onnx",
        HAND_FEATURE_DIM,
        "attention",
    )

    print("Exporting samyukta (attention)...")
    samyukta_classes = export(
        "checkpoints/samyukta_attention.pt",
        "web/models/samyukta.onnx",
        TWO_HAND_FEATURE_DIM,
        "attention_two_hand",
    )

    with open("web/models/classes.json", "w") as f:
        json.dump(
            {"asamyukta": asamyukta_classes, "samyukta": samyukta_classes},
            f, indent=2,
        )
    print("Saved class names to web/models/classes.json")


if __name__ == "__main__":
    main()

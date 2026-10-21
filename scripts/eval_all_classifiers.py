"""Combined eval for both deployed classifiers (asamyukta attention,
samyukta MLP — the two checkpoints actually exported to ONNX and shipped
to the browser). Outputs one JSON with per-class F1, top-1/top-3, macro/
weighted averages, and confusion matrices, plus a confusion matrix PNG
per model for the README.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.preprocess import build_asamyukta_dataset, build_samyukta_dataset
from src.models.classifier import AttentionClassifier, SamyuktaClassifier

OUT_DIR = Path("eval")
OUT_DIR.mkdir(exist_ok=True)


def top_k_accuracy(logits: np.ndarray, y_true: np.ndarray, k: int) -> float:
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float(np.mean([y in row for y, row in zip(y_true, topk)]))


def eval_model(name, checkpoint_path, model_cls, build_dataset_fn, figsize):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    model = model_cls(n_classes=len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data = build_dataset_fn()
    x_val = torch.from_numpy(data["x_val"]).float()
    y_val = data["y_val"]

    with torch.no_grad():
        logits = model(x_val).numpy()
    preds = logits.argmax(axis=1)

    report = classification_report(
        y_val, preds, target_names=class_names, digits=3, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_val, preds).tolist()

    metrics = {
        "checkpoint": checkpoint_path,
        "arch": ckpt["arch"],
        "n_classes": len(class_names),
        "n_val": int(len(y_val)),
        "class_names": class_names,
        "top1_accuracy": float(np.mean(preds == y_val)),
        "top3_accuracy": top_k_accuracy(logits, y_val, 3),
        "macro_f1": float(f1_score(y_val, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_val, preds, average="weighted", zero_division=0)),
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"],
            }
            for cls in class_names
        },
        "confusion_matrix": cm,
    }

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} confusion matrix ({ckpt['arch']})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    png_path = OUT_DIR / f"confusion_matrix_{name}.png"
    plt.savefig(png_path, dpi=120)
    plt.close(fig)

    print(f"{name}: top1={metrics['top1_accuracy']:.4f}  top3={metrics['top3_accuracy']:.4f}  "
          f"macro_f1={metrics['macro_f1']:.4f}  weighted_f1={metrics['weighted_f1']:.4f}")
    print(f"  saved {png_path}")
    return metrics


def main():
    results = {
        "asamyukta": eval_model(
            "asamyukta", "checkpoints/asamyukta_attention.pt",
            AttentionClassifier, build_asamyukta_dataset, (14, 12),
        ),
        "samyukta": eval_model(
            "samyukta", "checkpoints/samyukta_attention.pt",
            SamyuktaClassifier, build_samyukta_dataset, (10, 8),
        ),
    }
    out_path = OUT_DIR / "classifier_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

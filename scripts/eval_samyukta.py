"""Eval a samyukta checkpoint: classification report + confusion matrix."""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.preprocess import build_samyukta_dataset
from src.models.classifier import TwoHandMLPClassifier, SamyuktaClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/samyukta_attention.pt")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    arch = ckpt["arch"]

    model = (SamyuktaClassifier if arch == "attention" else TwoHandMLPClassifier)(
        n_classes=len(class_names)
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data = build_samyukta_dataset()
    x_val = torch.from_numpy(data["x_val"]).float()
    y_val = data["y_val"]

    with torch.no_grad():
        preds = model(x_val).argmax(dim=-1).numpy()

    print(classification_report(y_val, preds, target_names=class_names, digits=3))

    cm = confusion_matrix(y_val, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Samyukta confusion matrix ({arch})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out = f"confusion_matrix_samyukta_{arch}.png"
    plt.savefig(out, dpi=120)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

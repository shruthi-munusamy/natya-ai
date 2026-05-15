"""Quick EDA: class distribution + grid of sample mudras."""
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.loader import load_mudra_dataset


def main():
    ds = load_mudra_dataset()
    train = ds["train"]

    labels = [s["label"] for s in train]
    counts = Counter(labels)

    print(f"Total samples: {len(train)}")
    print(f"Num classes: {len(counts)}")
    print(f"Min samples per class: {min(counts.values())}")
    print(f"Max samples per class: {max(counts.values())}")
    print(f"Mean samples per class: {sum(counts.values()) / len(counts):.0f}")

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    seen, idx = set(), 0
    for sample in train:
        label = sample["label"]
        if label in seen or idx >= 9:
            continue
        seen.add(label)
        ax = axes[idx // 3, idx % 3]
        ax.imshow(sample["image"])
        ax.set_title(f"Label: {label}", fontsize=10)
        ax.axis("off")
        idx += 1

    plt.tight_layout()
    plt.savefig("sample_mudras.png", dpi=120)
    print("Saved sample_mudras.png")


if __name__ == "__main__":
    main()

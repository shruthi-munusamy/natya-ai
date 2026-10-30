"""Train asamyukta classifier — MLP baseline and attention model."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.preprocess import build_asamyukta_dataset, horizontal_flip, random_rotate
from src.models.classifier import MLPClassifier, AttentionClassifier


def train_one_model(model, data, epochs, batch_size, lr, device, flip_prob=0.5, rotate_prob=0.8):
    x_train = torch.from_numpy(data["x_train"]).float()
    y_train = torch.from_numpy(data["y_train"]).long()
    x_val = torch.from_numpy(data["x_val"]).float().to(device)
    y_val = torch.from_numpy(data["y_val"]).long().to(device)

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True,
    )
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb_np = xb.numpy()
            if flip_prob > 0:
                flip = np.random.rand(len(xb_np)) < flip_prob
                if flip.any():
                    xb_np[flip] = horizontal_flip(xb_np[flip])
            if rotate_prob > 0:
                rotate = np.random.rand(len(xb_np)) < rotate_prob
                if rotate.any():
                    xb_np[rotate] = random_rotate(xb_np[rotate])
            xb = torch.from_numpy(xb_np).to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_val)
            preds = logits.argmax(dim=-1)
            acc = (preds == y_val).float().mean().item()
            top3 = (logits.topk(3, dim=-1).indices == y_val.unsqueeze(-1)).any(dim=-1).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1:3d}/{epochs}  val_acc={acc:.4f}  top3={top3:.4f}")

    return best_acc, best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    data = build_asamyukta_dataset(seed=args.seed)
    print(f"Classes: {data['n_classes']}, train: {data['n_train']}, val: {data['n_val']}\n")

    Path("checkpoints").mkdir(exist_ok=True)

    print("[1/2] MLP baseline")
    mlp = MLPClassifier(n_classes=data["n_classes"])
    mlp_acc, mlp_state = train_one_model(mlp, data, args.epochs, args.batch_size, 1e-3, device)
    torch.save({"state_dict": mlp_state, "class_names": data["class_names"], "arch": "mlp"},
               "checkpoints/asamyukta_mlp.pt")
    print(f"MLP best val acc: {mlp_acc:.4f}\n")

    print("[2/2] Attention model")
    attn = AttentionClassifier(n_classes=data["n_classes"])
    attn_acc, attn_state = train_one_model(attn, data, args.epochs, args.batch_size, 1e-3, device)
    torch.save({"state_dict": attn_state, "class_names": data["class_names"], "arch": "attention"},
               "checkpoints/asamyukta_attention.pt")
    print(f"Attention best val acc: {attn_acc:.4f}\n")

    print(f"=== {data['n_classes']}-class asamyukta ===")
    print(f"MLP baseline:   {mlp_acc:.4f}")
    print(f"Attention:      {attn_acc:.4f}")
    print(f"Delta:          {(attn_acc - mlp_acc) * 100:+.2f} pp")


if __name__ == "__main__":
    main()

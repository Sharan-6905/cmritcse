"""Train a ResNet50 waste classifier with a stratified, seeded train/val/test split.

The best checkpoint is chosen on the validation set only; the test set is
scored exactly once at the end.
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EXTS = {".jpg", ".jpeg", ".png"}


class ImageList(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label


def scan(root: Path):
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    items = [(p, classes.index(d.name)) for d in root.iterdir() if d.is_dir()
             for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS]
    return classes, items


def evaluate(model, loader, device, criterion):
    model.eval()
    preds, labels, loss_sum = [], [], 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * len(y)
            preds += out.argmax(1).cpu().tolist()
            labels += y.cpu().tolist()
    acc = float(np.mean(np.array(preds) == np.array(labels)))
    return loss_sum / len(labels), acc, labels, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="weights")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(exist_ok=True)

    classes, items = scan(Path(args.data))
    labels = [l for _, l in items]
    train_i, rest_i = train_test_split(range(len(items)), test_size=0.30, stratify=labels, random_state=args.seed)
    val_i, test_i = train_test_split(rest_i, test_size=0.50, stratify=[labels[i] for i in rest_i], random_state=args.seed)
    pick = lambda idx: [items[i] for i in idx]
    print(f"classes={classes} train={len(train_i)} val={len(val_i)} test={len(test_i)} device={device}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20), transforms.ColorJitter(0.3, 0.3, 0.2),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    mk = lambda idx, tf, shuf: DataLoader(ImageList(pick(idx), tf), batch_size=args.batch,
                                          shuffle=shuf, num_workers=2, persistent_workers=True)
    train_dl, val_dl, test_dl = mk(train_i, train_tf, True), mk(val_i, eval_tf, False), mk(test_i, eval_tf, False)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)

    counts = np.bincount([labels[i] for i in train_i], minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history, best_acc, best_epoch, patience = [], 0.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            run_loss += loss.item() * len(y)
        scheduler.step()
        val_loss, val_acc, _, _ = evaluate(model, val_dl, device, criterion)
        history.append({"epoch": epoch, "train_loss": run_loss / len(train_i), "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch}: train_loss={history[-1]['train_loss']:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_epoch, patience = val_acc, epoch, 0
            torch.save(model.state_dict(), out / "best_model.pth")
        else:
            patience += 1
            if patience >= 5:
                print("early stopping"); break

    model.load_state_dict(torch.load(out / "best_model.pth", map_location=device))
    _, test_acc, y_true, y_pred = evaluate(model, test_dl, device, criterion)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=classes))

    (out / "classes.json").write_text(json.dumps(classes))
    (out / "metrics.json").write_text(json.dumps({
        "split": {"train": len(train_i), "val": len(val_i), "test": len(test_i)},
        "best_val_acc": best_acc, "best_epoch": best_epoch, "test_acc": test_acc,
        "per_class": report, "history": history}, indent=2))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=classes).plot(xticks_rotation=45)
    plt.title(f"Test confusion matrix (acc {test_acc:.3f})"); plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    print(f"best_val_acc={best_acc:.4f} (epoch {best_epoch}) test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()

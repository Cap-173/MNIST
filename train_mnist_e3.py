import argparse
import csv
import json
import random
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.models.resnet import BasicBlock
 
 
# Residual blocks from Torchvision, with a stem and classifier sized for MNIST.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            BasicBlock(32, 32),
            BasicBlock(32, 64, stride=2, downsample=nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64),
            )),
            BasicBlock(64, 64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return F.log_softmax(self.classifier(self.features(x)), dim=1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 
 
def evaluate(model, loader, device, collect=False):
    model.eval()
    total_loss = correct = count = 0
    ys, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.nll_loss(
                logits, y, reduction="sum"
            ).item()
            prediction = logits.argmax(dim=1)
            correct += (prediction == y).sum().item()
            count += y.numel()
            if collect:
                ys.extend(y.cpu().tolist())
                preds.extend(prediction.cpu().tolist())
    return total_loss / count, correct / count, ys, preds
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.7)
    args = parser.parse_args()
    set_seed(args.seed)
 
    out = Path("outputs_e3")
    out.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("PyTorch:", torch.__version__)
 
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST(
        "data", train=True, download=True, transform=tfm
    )
    # No access to the official test split during model selection.
    generator = torch.Generator().manual_seed(args.seed)
    train_indices, val_set = random_split(
        full_train, [54000, 6000], generator=generator
    )
    train_tfm = transforms.Compose([
        transforms.RandomAffine(
            degrees=10, translate=(0.07, 0.07), scale=(0.95, 1.05),
            interpolation=transforms.InterpolationMode.BILINEAR, fill=0,
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    augmented_train = datasets.MNIST(
        "data", train=True, download=False, transform=train_tfm
    )
    train_set = Subset(augmented_train, train_indices.indices)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_set, batch_size=512, shuffle=False, num_workers=0
    )
    print("Train:", len(train_set), "Validation:", len(val_set))
 
    model = Net().to(device)
    opt = optim.Adadelta(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.gamma)
    best_acc, best_loss, best_epoch = -1.0, float("inf"), 0
    history = []
 
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = count = correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.nll_loss(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            count += y.numel()
        train_loss, train_acc = total_loss / count, correct / count
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        row = dict(epoch=epoch, train_loss=train_loss,
                   train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
        history.append(row)
        print(f"Epoch {epoch:02d} | train {train_acc:.4%} "
              f"val {val_acc:.4%} | val_loss {val_loss:.4f}")
        if val_acc > best_acc or (
            val_acc == best_acc and val_loss < best_loss
        ):
            best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
            torch.save(model.state_dict(), out / "best_mnist_cnn.pt")
        sched.step()
 
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
 
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ep = [x["epoch"] for x in history]
    for name in ("train_loss", "val_loss"):
        ax[0].plot(ep, [x[name] for x in history], label=name)
    for name in ("train_acc", "val_acc"):
        ax[1].plot(ep, [x[name] for x in history], label=name)
    ax[0].set_title("Loss")
    ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("Epoch")
        a.legend()
        a.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=180)
    plt.close(fig)
 
    result = {
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "best_val_loss": best_loss,
        "best_val_errors": round((1 - best_acc) * len(val_set)),
        "model": "mnist_residual_cnn",
        "parameters": sum(p.numel() for p in model.parameters()),
        "seed": args.seed,
        "epochs": args.epochs,
    }
    (out / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"Best validation: epoch {best_epoch}, {best_acc:.4%}, "
          f"{result['best_val_errors']} errors out of {len(val_set)}")


if __name__ == "__main__":
    main()

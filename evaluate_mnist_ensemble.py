"""Evaluate the fixed equal-weight E1/E2 ensemble on validation or test."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from train_mnist_e1 import Net, evaluate


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, images):
        log_probs = torch.stack([model(images) for model in self.models])
        return torch.logsumexp(log_probs, dim=0) - math.log(len(self.models))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(
        "data", train=not args.test, download=False, transform=transform
    )
    if not args.test:
        _, dataset = random_split(
            dataset, [54000, 6000], generator=torch.Generator().manual_seed(42)
        )

    models = []
    for run in ("outputs_e1", "outputs_e2"):
        model = Net().to(device)
        state = torch.load(Path(run) / "best_mnist_cnn.pt",
                           map_location=device, weights_only=True)
        model.load_state_dict(state)
        models.append(model)
    loader = DataLoader(dataset, batch_size=512)
    loss, accuracy, labels, predictions = evaluate(
        Ensemble(models), loader, device, collect=True
    )
    result = dict(
        split="test" if args.test else "validation",
        accuracy=accuracy, loss=loss, count=len(dataset),
        errors=sum(label != prediction for label, prediction in zip(labels, predictions)),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

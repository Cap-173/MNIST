"""Select E4 or an equal-weight E3/E4 ensemble using validation only."""
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from train_mnist_e3 import Net, evaluate


class Ensemble(nn.Module):
    def __init__(self, e3, e4):
        super().__init__()
        self.e3 = e3
        self.e4 = e4

    def forward(self, x):
        return torch.logaddexp(self.e3(x), self.e4(x)) - math.log(2)


def score(model, dataset, device):
    loss, accuracy, labels, predictions = evaluate(
        model, DataLoader(dataset, batch_size=512), device, collect=True
    )
    return dict(
        accuracy=accuracy, loss=loss, count=len(labels),
        errors=sum(label != prediction for label, prediction in zip(labels, predictions)),
    )


def main():
    out = Path("outputs_e4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
    ])
    data = datasets.MNIST("data", train=True, download=False, transform=transform)
    _, val = random_split(data, [54000, 6000], generator=torch.Generator().manual_seed(42))
    models = {}
    for run in ("e3", "e4"):
        model = Net().to(device)
        model.load_state_dict(torch.load(
            Path(f"outputs_{run}") / "best_mnist_cnn.pt",
            map_location=device, weights_only=True,
        ))
        models[run] = model
    models["ensemble"] = Ensemble(models["e3"], models["e4"])
    validation = {name: score(model, val, device) for name, model in models.items()}
    winner = min(validation, key=lambda name: validation[name]["errors"])
    result = dict(validation=validation, selected=winner)
    print(json.dumps(result, indent=2), flush=True)
    if winner != "e3":
        test = datasets.MNIST("data", train=False, download=False, transform=transform)
        result["test"] = score(models[winner], test, device)
        print("Final test:", json.dumps(result["test"]), flush=True)
    with (out / "comparison.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)


if __name__ == "__main__":
    main()

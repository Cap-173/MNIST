"""用指定的 E9 权重评估官方 MNIST 测试集。"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_mnist_e9 import Net, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="E9 权重文件路径")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"找不到权重文件：{args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 与 E9 验证预处理一致，只做张量转换和归一化。
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_set = datasets.MNIST("data", train=False, download=False, transform=transform)
    loader = DataLoader(test_set, batch_size=512, shuffle=False)

    model = Net().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    loss, accuracy, labels, predictions = evaluate(model, loader, device, collect=True)
    errors = sum(label != pred for label, pred in zip(labels, predictions))
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test accuracy: {accuracy:.4%}")
    print(f"Test loss: {loss:.6f}")
    print(f"Test errors: {errors} / {len(test_set)}")


if __name__ == "__main__":
    main()

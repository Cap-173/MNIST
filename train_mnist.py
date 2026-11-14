# E0 两层 CNN 基线
# 解析命令行参数；CSV 保存逐轮指标，JSON 保存汇总，Path 用于拼接文件路径。
import argparse
import csv
import json
import random
from pathlib import Path
 
# 绘制训练曲线；PyTorch 负责张量、网络、自动求导和优化。
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
 
 
# The network structure is adapted from pytorch/examples/mnist/main.py.
# 定义网络层与前向路径。nn.Module 会登记子层的参数，供优化器和权重保存使用。
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 两层卷积都不填充、步长为 1。通道从 1 到 32 再到 64。
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
 
    def forward(self, x):
        # 第一层卷积与 ReLU 输出 [B, 32, 26, 26]。
        x = F.relu(self.conv1(x))
        # 第二层卷积与 ReLU 输出 [B, 64, 24, 24]。
        x = F.relu(self.conv2(x))
        # 用 2×2 最大池化将空间大小降到 12×12。
        x = F.max_pool2d(x, 2)
        # 训练时随机置零一部分特征，缓解对个别特征的依赖。
        x = self.dropout1(x)
        # 从第 1 维开始展平，保留批次维，得到 [B, 9216]。
        x = torch.flatten(x, 1)
        # 将 9216 维映射到 128 维，再接 Dropout 和十分类输出层。
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        # 输出十个类别的对数概率，与 F.nll_loss 搭配使用。
        return F.log_softmax(x, dim=1)
 
 
# 给 Python、NumPy、PyTorch CPU 与 CUDA 设置随机种子，方便复现实验。GPU 算子仍可能存在非确定性。
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 
 
# 在验证集或测试集上计算损失和准确率。collect=True 时额外收集真实标签与预测类别。
def evaluate(model, loader, device, collect=False):
    # 切到评估模式，使 Dropout 停用；有 BatchNorm 的模型会使用训练期间记录的统计量。
    model.eval()
    total_loss = correct = count = 0
    ys, preds = [], []
    # 评估只做前向。关闭梯度记录可降低显存占用，也避免构建反向传播图。
    with torch.no_grad():
        for x, y in loader:
            # 当前批图像和整数类别标签都移动到模型所在设备。
            x, y = x.to(device), y.to(device)
            # 逐批预测。输出形状为 [B, 10]，B 是当前批量大小。
            logits = model(x)
            # 用 reduction='sum' 累加批内损失，最后除以总样本数，使最后一个小批次也按样本数计权。
            total_loss += F.nll_loss(
                logits, y, reduction="sum"
            ).item()
            # 沿类别维取最大值的位置，得到每张图的预测数字，形状为 [B]。
            prediction = logits.argmax(dim=1)
            correct += (prediction == y).sum().item()
            count += y.numel()
            # 只在需要混淆矩阵或逐样本分析时保存标签。移回 CPU 后转成普通列表。
            if collect:
                ys.extend(y.cpu().tolist())
                preds.extend(prediction.cpu().tolist())
    return total_loss / count, correct / count, ys, preds
 
 
# 训练入口。依次准备参数、设备、数据、模型，最后保存选定权重和指标。
def main():
    parser = argparse.ArgumentParser()
    # 命令行参数可以覆盖默认训练轮数、批量大小和训练种子。
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.7)
    args = parser.parse_args()
    set_seed(args.seed)
 
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    # CUDA 可用时选择 GPU，否则使用 CPU。模型和同批数据必须位于同一设备。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("PyTorch:", torch.__version__)
 
    # ToTensor 把灰度图变成 [1, 28, 28]，像素缩放到 [0, 1]。Normalize 按 (x - 0.1307) / 0.3081 标准化。
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 读取官方训练部分的 60,000 张图，后面从中抽取验证集。
    full_train = datasets.MNIST(
        "data", train=True, download=True, transform=tfm
    )
    # No access to the official test split during model selection.
    # 按 --seed 固定这次划分，60,000 张图分为 54,000 张训练与 6,000 张验证。
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(
        full_train, [54000, 6000], generator=generator
    )
    # 训练集按批读取并打乱顺序。默认批量为 64；CUDA 时启用页锁定内存，便于数据传输。
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda"
    )
    # 验证集固定顺序，批量可大于训练集，因为评估不用保留反向传播图。
    val_loader = DataLoader(
        val_set, batch_size=512, shuffle=False, num_workers=0
    )
    print("Train:", len(train_set), "Validation:", len(val_set))
 
    # 实例化模型并移到计算设备，再将其可训练参数交给优化器。
    model = Net().to(device)
    # E0 用 Adadelta，根据梯度统计调整参数更新量。
    opt = optim.Adadelta(model.parameters(), lr=args.lr)
    # StepLR 每轮结束将学习率乘以 gamma，默认是 0.7。
    sched = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.gamma)
    # 分别记录最佳验证准确率、对应损失和轮次。history 保存每轮统计量。
    best_acc, best_loss, best_epoch = -1.0, float("inf"), 0
    history = []
 
    # 一个 epoch 遍历一次训练集，随后完整验证一次。
    for epoch in range(1, args.epochs + 1):
        # 重新启用训练模式。上一轮 evaluate 已切到评估模式。
        model.train()
        total_loss = count = correct = 0
        for x, y in train_loader:
            # 当前批图像和整数类别标签都移动到模型所在设备。
            x, y = x.to(device), y.to(device)
            # 清除上一批的梯度。PyTorch 的梯度默认会累加。
            opt.zero_grad()
            # 逐批预测。输出形状为 [B, 10]，B 是当前批量大小。
            logits = model(x)
            # 计算当前批的平均分类损失。标签 y 是取值 0～9 的类别编号。
            loss = F.nll_loss(logits, y)
            # 沿计算图求损失对参数的梯度，随后由 opt.step() 更新参数。
            loss.backward()
            opt.step()
            # 将批均值乘以样本数，累加得到可用于计算整轮均值的损失总和。
            total_loss += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            count += y.numel()
        # 除以实际处理的样本总数，得到整轮训练指标。
        train_loss, train_acc = total_loss / count, correct / count
        # 用更新后的模型跑验证集。验证指标用于选权重。
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        row = dict(epoch=epoch, train_loss=train_loss,
                   train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
        history.append(row)
        print(f"Epoch {epoch:02d} | train {train_acc:.4%} "
              f"val {val_acc:.4%} | val_loss {val_loss:.4f}")
        # 优先选验证准确率更高的轮次；准确率相同时，选验证损失更低的轮次。
        if val_acc > best_acc or (
            val_acc == best_acc and val_loss < best_loss
        ):
            best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
            # 保存参数和缓冲区。加载时需要先构建相同网络结构。
            torch.save(model.state_dict(), out / "best_mnist_cnn.pt")
        # 在一轮训练和验证完成后推进调度器，为下一轮设置学习率。
        sched.step()
 
    # 保存每轮指标，便于复核、绘图和对比不同运行。
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
 
    # 左图画损失，右图画准确率。横轴是训练轮次。
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
 
    # Final test: performed only after the checkpoint is chosen.
    # 使用官方 10,000 张测试图做最终评价。
    test_set = datasets.MNIST(
        "data", train=False, download=True, transform=tfm
    )
    test_loader = DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=0
    )
    # 恢复按验证集选出的最佳权重，测试该轮模型。
    model.load_state_dict(
        torch.load(out / "best_mnist_cnn.pt", map_location=device,
                   weights_only=True)
    )
    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, device, collect=True
    )
    # 混淆矩阵的行是真实数字，列是预测数字。对角线记录正确分类数量。
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))
    disp.plot(cmap="Blues", values_format="d", colorbar=False)
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=180)
    plt.close()
    # 将本次配置和最佳结果写入 JSON，便于汇总。
    result = {"best_epoch": best_epoch, "best_val_acc": best_acc,
              "test_acc": test_acc, "test_loss": test_loss,
              "seed": args.seed, "epochs": args.epochs}
    (out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Final independent test accuracy:", f"{test_acc:.4%}")
 
 
# 直接执行脚本时开始训练；作为模块导入时可以复用 Net 和 evaluate。
if __name__ == "__main__":
    main()

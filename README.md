# MNIST 手写数字识别

用 PyTorch 比较仿射增强、残差结构、优化配置、池化位置、EMA 与标签平滑。最终采用 E9 无标签平滑配置，三个训练种子的官方测试准确率平均为 **99.6733%**。E7 的最高单次成绩是 **99.72%**。

完整的实验设计、曲线和错分分析见 [实验总结](MNIST实验总结.md)。逐轮记录存放在 `results/`，汇报材料见 [汇报讲稿](MNIST汇报讲稿.md)。

## 复现实验

实验环境为 Python 3.13、PyTorch 2.15.0、Torchvision 0.29.2 和 RTX 4060 Laptop GPU。请使用与设备匹配的 PyTorch 构建。[PyTorch 安装说明](https://pytorch.org/get-started/locally/)

在仓库根目录运行：

```powershell
python -m pip install -r requirements.txt
python train_mnist_e9.py --seed 42 --label-smoothing 0.0 --output-dir outputs_e9_local
python evaluate_mnist_e9.py --checkpoint outputs_e9_local/best_mnist_cnn.pt
```

训练脚本会下载 MNIST。每次训练请指定新的输出目录。将训练种子换成 43 和 44 可复跑另外两组，将标签平滑改为 0.05 可复跑对应对照。数据划分种子固定为 42。

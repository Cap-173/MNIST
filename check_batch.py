import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda:0")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
dataset = datasets.MNIST(
    "data", train=True, download=False, transform=transform
)
x, y = next(iter(DataLoader(dataset, batch_size=64)))
x, y = x.to(device), y.to(device)

# 沿用手册中的 CNN 结构，写成顺序执行的形式
model = nn.Sequential(
    nn.Conv2d(1, 32, 3), nn.ReLU(),
    nn.Conv2d(32, 64, 3), nn.ReLU(),
    nn.MaxPool2d(2), nn.Dropout(0.25),
    nn.Flatten(),
    nn.Linear(9216, 128), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(128, 10), nn.LogSoftmax(dim=1),
).to(device)

optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
optimizer.zero_grad()
log_probs = model(x)
loss = nn.NLLLoss()(log_probs, y)
loss.backward()
optimizer.step()

print("计算设备", log_probs.device)
print("输出形状", log_probs.shape)
print("损失", loss.item())
"""
神经网络训练
input: 70个特征，1个目标值
"""

import statistics

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from network import SimpleNN

# 定义神经网络模型


df = pd.read_csv("output.csv")

input_data = df.iloc[:, :70].values  # 前70列作为输入特征


target_data = df.iloc[:, -1:].values  # 最后列作为目标值


# 将数据转换为PyTorch张量
input_data = torch.tensor(input_data, dtype=torch.float32)
target_data = torch.tensor(target_data, dtype=torch.float32)


# 创建数据集
dataset = TensorDataset(input_data, target_data)

# 创建数据加载器
BATCH_SIZE = 128
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# 初始化模型、损失函数和优化器
model = SimpleNN()
criterion = nn.L1Loss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
# 学习率调度器
scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

# 训练参数
NUM_EPOCHS = 10000
recent20loss = [0] * 20


# 开始训练
best_loss = float(1000.0)

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for inputs, targets in data_loader:
        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    scheduler.step(running_loss)
    recent20loss.append(running_loss)
    recent20loss.pop(0)
    variance = statistics.variance(recent20loss)
    if epoch % 7 == 0:
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}], Loss: {running_loss / len(data_loader):.4f}",
            end="     ",
        )
        print(variance)

    # if variance < 0.3:
    #     torch.save(model.state_dict(), 'model'+'variance' +
    #                str(variance)+'epoch'+str(epoch)+'loss'+str(running_loss)+'.pth')

    if running_loss < best_loss:
        best_loss = running_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model  " + str(best_loss / len(data_loader)))

    if running_loss < 0.05:
        break


# 保存模型的状态字典
torch.save(
    model.state_dict(),
    "model"
    + "variance"
    + str(variance)
    + "epoch"
    + str(epoch)
    + "loss"
    + str(running_loss)
    + ".pth",
)

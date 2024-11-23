"""
使用训练好的模型进行预测

"""


import pandas as pd
import torch
from network import SimpleNN

# 实例化模型
model = SimpleNN()

# 加载模型的状态字典
model.load_state_dict(torch.load("best_model.pth", weights_only=True))

# 将模型设置为评估模式
model.eval()


# 读取新的输入数据
column_names = [f"feature_{i}" for i in range(70)] + ["target1", "target2"]
new_df = pd.read_csv("output.csv", header=None, names=column_names)


# 提取输入特征
new_input_data = new_df.iloc[:, :70].values

# 将数据转换为PyTorch张量
new_input_data = torch.tensor(new_input_data, dtype=torch.float32)

# 进行前向传播
with torch.no_grad():  # 关闭梯度计算
    predictions = model(new_input_data, training=False)

# 将预测结果转换为NumPy数组（如果需要）
predictions = predictions.numpy()

new_df["prediction1"] = predictions[:, 0]
# new_df["prediction2"]=predictions[:,1]

# 保存更新后的DataFrame到output.csv
new_df.to_csv("outputwp.csv", index=False)

# 打印更新后的DataFrame
print(new_df)

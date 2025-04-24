# 开发时间：2024-08-15 14:26
# 开发时间：2024-08-15 13:50
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os


# 加载数据
class CSVDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []

        for label in range(10):
            folder_path = os.path.join(root_dir, str(label))
            for filename in os.listdir(folder_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(file_path, header=None)
                    feature = df.iloc[:, 1].values  # 提取第二列
                    self.data.append((feature, feature))  # 两个分支使用相同的特征
                    self.labels.append(label)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature1, feature2 = self.data[idx]
        label = self.labels[idx]
        return (torch.tensor(feature1, dtype=torch.float32),
                torch.tensor(feature2, dtype=torch.float32)), torch.tensor(label, dtype=torch.long)


# 预处理数据
root_dir = 'fault_zone'
dataset = CSVDataset(root_dir)

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


class LSTMBranch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMBranch, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 的形状应该是 (batch_size, seq_len, input_size)
        # 如果输入是 (batch_size, input_size)，你需要调整它的形状
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 将其调整为 (batch_size, 1, input_size)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x)  # LSTM 输出 (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]  # 取最后一个时间步的输出 (batch_size, hidden_size)
        out = self.fc(out)
        out = self.relu(out)
        return out



class DualBranchLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(DualBranchLSTMModel, self).__init__()
        self.branch1 = LSTMBranch(input_size, hidden_size, num_layers)
        self.branch2 = LSTMBranch(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(128, num_classes)  # 两个分支的输出拼接后送入全连接层

    def forward(self, x1, x2):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        out = torch.cat((out1, out2), dim=1)
        out = self.fc(out)
        return out


# 初始化模型
input_size = 241  # 输入特征数量
hidden_size = 128
num_layers = 2
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualBranchLSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
total_epochs = 100
for epoch in range(total_epochs):
    model.train()
    running_loss = 0.0
    for i, ((inputs1, inputs2), labels) in enumerate(train_loader):
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs1, inputs2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{total_epochs}], Loss: {running_loss / len(train_loader)}")

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for (inputs1, inputs2), labels in test_loader:
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

        outputs = model(inputs1, inputs2)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the model on the test dataset: {(correct / total) * 100:.2f}%")

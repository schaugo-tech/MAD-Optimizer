# 开发时间：2024-08-19 22:04
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
                    feature = df.iloc[:, 0].values  # 提取第1列
                    self.data.append(feature)
                    self.labels.append(label)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.data[idx]
        label = self.labels[idx]
        return (torch.tensor(feature, dtype=torch.float32),
                torch.tensor(feature, dtype=torch.float32)), torch.tensor(label, dtype=torch.long)

# 数据集分割
def split_dataset(dataset, test_size=0.2):
    data_len = len(dataset)
    indices = list(range(data_len))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=32)

    train_data = torch.utils.data.Subset(dataset, train_indices)
    test_data = torch.utils.data.Subset(dataset, test_indices)
    return train_data, test_data

# 预处理数据
root_dir = 'fault_zone'
dataset = CSVDataset(root_dir)
train_data, test_data = split_dataset(dataset, test_size=0.2)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# CNN 分支
class CNNBranch(nn.Module):
    def __init__(self, input_size):
        super(CNNBranch, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(32 * (input_size // 4), 64)  # 调整输入大小
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        x = self.relu(x)
        return x

# LSTM 分支
class LSTMBranch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMBranch, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 将其调整为 (batch_size, 1, input_size)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.relu(out)
        return out

# 双分支模型：一个分支是 CNN，另一个分支是 LSTM
class DualBranchCNNLSTMModel(nn.Module):
    def __init__(self, cnn_input_size, lstm_input_size, lstm_hidden_size, lstm_num_layers, num_classes):
        super(DualBranchCNNLSTMModel, self).__init__()
        self.cnn_branch = CNNBranch(cnn_input_size)
        self.lstm_branch = LSTMBranch(lstm_input_size, lstm_hidden_size, lstm_num_layers)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x1, x2):
        out1 = self.cnn_branch(x1)
        out2 = self.lstm_branch(x2)
        out = torch.cat((out1, out2), dim=1)
        out = self.fc(out)
        return out

# 初始化模型
input_size = 22 # 输入特征数量
lstm_hidden_size = 128
lstm_num_layers = 2
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualBranchCNNLSTMModel(input_size, input_size, lstm_hidden_size, lstm_num_layers, num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
total_epochs = 200
for epoch in range(total_epochs):
    model.train()
    running_loss = 0.0
    for i, ((inputs1, inputs2), labels) in enumerate(train_loader):
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

        inputs1 = inputs1.unsqueeze(1)
        inputs2 = inputs2.unsqueeze(-1)
        inputs2 = inputs2.permute(0, 2, 1)

        optimizer.zero_grad()

        outputs = model(inputs1, inputs2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{total_epochs}], Loss: {running_loss / len(train_loader)}")

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for (inputs1, inputs2), labels in test_loader:
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
        inputs1 = inputs1.unsqueeze(1)
        inputs2 = inputs2.unsqueeze(-1)
        inputs2 = inputs2.permute(0, 2, 1)
        outputs = model(inputs1, inputs2)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the model on the test dataset: {(correct / total) * 100:.2f}%")

# 保存模型
torch.save(model.state_dict(), 'dual_branch_cnn_lstm_model.pth')

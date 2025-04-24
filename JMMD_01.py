import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os

# 自定义数据集类，用于加载源域数据
class CSVDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []

        for label in range(10):  # 源域有10个文件夹，每个文件夹表示一个类别
            folder_path = os.path.join(root_dir, str(label))
            for filename in os.listdir(folder_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(file_path, header=None)
                    feature = df.iloc[:, 0].values  # 提取第1列作为特征
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

# 自定义数据集类，用于加载目标域数据
class TargetCSVDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []

        for label in range(6):  # 目标域有6个文件夹，每个文件夹表示一个类别
            folder_path = os.path.join(root_dir, str(label))
            for filename in os.listdir(folder_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(file_path, header=None)
                    feature = df.iloc[:, 0].values  # 提取第1列作为特征
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

# CNN 分支
class CNNBranch(nn.Module):
    def __init__(self, input_size):
        super(CNNBranch, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(32 * (input_size // 2 // 2), 64)
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

# J-MMD 损失计算
class DualBranchCNNLSTMModelWithJMMD(nn.Module):
    def __init__(self, cnn_input_size, lstm_input_size, lstm_hidden_size, lstm_num_layers, num_classes):
        super(DualBranchCNNLSTMModelWithJMMD, self).__init__()
        self.cnn_branch = CNNBranch(cnn_input_size)
        self.lstm_branch = LSTMBranch(lstm_input_size, lstm_hidden_size, lstm_num_layers)
        self.fc = nn.Linear(128, num_classes)  # 分类层

    def forward(self, x1, x2):
        out1 = self.cnn_branch(x1)
        out2 = self.lstm_branch(x2)
        out = torch.cat((out1, out2), dim=1)
        class_output = self.fc(out)
        return class_output, out  # 返回特征用于 J-MMD 损失计算

    def jmmd_loss(self, src_features, tgt_features):
        # 计算源域特征和目标域特征的均值
        src_mean = torch.mean(src_features, dim=0)
        tgt_mean = torch.mean(tgt_features, dim=0)

        # 计算 J-MMD 损失（例如，使用均方误差）
        jmmd_loss = torch.mean((src_mean - tgt_mean) ** 2)
        return jmmd_loss

# 训练和测试设置
input_size = 22  # 输入特征数量
lstm_hidden_size = 128
lstm_num_layers = 2
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualBranchCNNLSTMModelWithJMMD(input_size, input_size, lstm_hidden_size, lstm_num_layers, num_classes).to(device)

# 定义损失函数和优化器
criterion_class = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 加载源域数据
root_dir_source = 'fault_zone'
source_dataset = CSVDataset(root_dir_source)
train_data, _ = train_test_split(source_dataset, test_size=0.2, random_state=56)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 训练源域模型
total_epochs = 200
for epoch in range(total_epochs):
    model.train()
    running_loss = 0.0
    for i, ((inputs1, inputs2), labels) in enumerate(train_loader):
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

        inputs1 = inputs1.unsqueeze(1)  # 增加通道维度
        inputs2 = inputs2.unsqueeze(-1).permute(0, 2, 1)  # 调整形状 (batch_size, seq_len, input_size)

        optimizer.zero_grad()

        class_output, src_features = model(inputs1, inputs2)
        class_loss = criterion_class(class_output, labels)

        # 使用源域特征计算 J-MMD 损失
        src_features = src_features.detach()  # 只需要计算 J-MMD 损失
        tgt_features = src_features.clone().detach()  # 目标域特征（这里只是一个示例，需要替换为实际的目标域特征）
        jmmd_loss = model.jmmd_loss(src_features, tgt_features)

        total_loss = class_loss + jmmd_loss
        total_loss.backward()
        optimizer.step()

        running_loss += class_loss.item()

    # print(f"Epoch [{epoch + 1}/{total_epochs}], Classification Loss: {running_loss / len(train_loader)}")

# 加载目标域数据
root_dir_target = 'gob_zone'
target_dataset = TargetCSVDataset(root_dir_target)
target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)

# 微调模型在目标域数据上
for epoch in range(total_epochs):
    model.train()
    running_loss = 0.0
    for i, ((inputs1, inputs2), labels) in enumerate(target_loader):
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

        inputs1 = inputs1.unsqueeze(1)  # 增加通道维度
        inputs2 = inputs2.unsqueeze(-1).permute(0, 2, 1)  # 调整形状 (batch_size, seq_len, input_size)

        optimizer.zero_grad()

        class_output, tgt_features = model(inputs1, inputs2)
        class_loss = criterion_class(class_output, labels)

        # 计算 J-MMD 损失
        src_features = torch.zeros_like(tgt_features)  # 源域特征的均值，用于计算 J-MMD 损失
        jmmd_loss = model.jmmd_loss(src_features, tgt_features)

        total_loss = class_loss + jmmd_loss
        total_loss.backward()
        optimizer.step()

        running_loss += class_loss.item()

    # print(f"Epoch [{epoch + 1}/{total_epochs}], Classification Loss: {running_loss / len(target_loader)}")
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for (inputs1, inputs2), labels in target_loader:
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            inputs1 = inputs1.unsqueeze(1)  # 增加通道维度
            inputs2 = inputs2.unsqueeze(-1).permute(0, 2, 1)  # 调整形状
            # class_output, _ = model(inputs1, inputs2, reverse_gradient=False)
            class_output, _ = model(inputs1, inputs2)
            _, predicted = torch.max(class_output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100. * correct / total
    print(f"Epoch [{epoch + 1}/{total_epochs}], Target Domain Test Accuracy: {test_accuracy:.2f}%")
    with open('JMMD_01.txt', 'a') as f:
        f.write("第{}轮训练的准确率：{}\n".format(epoch + 1, str(test_accuracy)[0:5]+'%'))

    # 将模型设置回训练模式
    model.train()
    # 测试模型的性能
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for (inputs1, inputs2), labels in target_loader:
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
        inputs1 = inputs1.unsqueeze(1)  # 增加通道维度
        inputs2 = inputs2.unsqueeze(-1).permute(0, 2, 1)  # 调整形状 (batch_size, seq_len, input_size)
        class_output, _ = model(inputs1, inputs2)
        _, predicted = torch.max(class_output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"模型在目标域测试数据上的准确率: {(correct / total) * 100:.2f}%")

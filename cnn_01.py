# 开发时间：2024-08-15 13:00
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
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# 预处理数据
root_dir = 'fault_zone'
dataset = CSVDataset(root_dir)

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


class CNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * (input_size // 2 // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 初始化模型
input_size = 22  # 输入特征数量
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel(input_size, num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
total_epochs = 200
for epoch in range(total_epochs):
    model.train()
    running_loss = 0.0

    total_train_accuracy = 0
    total_train_loss = 0
    train_data_size = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        inputs = inputs.unsqueeze(1)  # 增加通道维度
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        train_accuracy = (outputs.argmax(1) == labels).sum()
        total_train_accuracy = total_train_accuracy + train_accuracy
        total_train_loss = total_train_loss + loss
        train_data_size = train_data_size + labels.size(0)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # print(f"Epoch [{epoch+1}/{total_epochs}], Loss: {running_loss / len(train_loader)}")
    print("第{}轮训练的准确率：{}".format(epoch + 1, total_train_accuracy / train_data_size))
    with open('cnn_01.txt', 'a') as f:
        f.write("第{}轮训练的准确率：{}\n".format(epoch + 1, total_train_accuracy / train_data_size))

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(1)  # 增加通道维度
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the model on the test dataset: {(correct / total) * 100:.2f}%")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os

# Custom dataset class for source domain
class CSVDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []

        for label in range(10):  # Source domain has 10 folders, each representing a class
            folder_path = os.path.join(root_dir, str(label))
            for filename in os.listdir(folder_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(file_path, header=None)
                    feature = df.iloc[:, 0].values  # Extract the first column as feature
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

# Custom dataset class for target domain
class TargetCSVDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []

        for label in range(6):  # Target domain has 6 folders, each representing a class
            folder_path = os.path.join(root_dir, str(label))
            for filename in os.listdir(folder_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(file_path, header=None)
                    feature = df.iloc[:, 0].values  # Extract the first column as feature
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

# CNN Branch
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
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = self.relu(x)
        return x

# LSTM Branch
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
            x = x.unsqueeze(1)  # Adjust to (batch_size, 1, input_size)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x)  # LSTM output (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]  # Take the last time step (batch_size, hidden_size)
        out = self.fc(out)
        out = self.relu(out)
        return out

# MMD Loss Calculation
class DualBranchCNNLSTMModelWithMMD(nn.Module):
    def __init__(self, cnn_input_size, lstm_input_size, lstm_hidden_size, lstm_num_layers, num_classes):
        super(DualBranchCNNLSTMModelWithMMD, self).__init__()
        self.cnn_branch = CNNBranch(cnn_input_size)
        self.lstm_branch = LSTMBranch(lstm_input_size, lstm_hidden_size, lstm_num_layers)
        self.fc = nn.Linear(128, num_classes)  # Classification layer

    def forward(self, x1, x2):
        out1 = self.cnn_branch(x1)
        out2 = self.lstm_branch(x2)
        out = torch.cat((out1, out2), dim=1)
        class_output = self.fc(out)
        return class_output, out  # Return features for MMD loss calculation

    def mmd_loss(self, src_features, tgt_features):
        # Compute the MMD loss
        src_mean = torch.mean(src_features, dim=0)
        tgt_mean = torch.mean(tgt_features, dim=0)

        # Compute MMD loss using squared distance
        mmd_loss = torch.mean((src_mean - tgt_mean) ** 2)
        return mmd_loss

# Training and Testing Setup
input_size = 22  # Number of input features
lstm_hidden_size = 256
lstm_num_layers = 2
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualBranchCNNLSTMModelWithMMD(input_size, input_size, lstm_hidden_size, lstm_num_layers, num_classes).to(device)

# Define loss functions and optimizer
criterion_class = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load source domain data
root_dir_source = 'fault_zone'
source_dataset = CSVDataset(root_dir_source)
train_data, _ = train_test_split(source_dataset, test_size=0.2, random_state=56)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Train the source domain model
total_epochs = 200
for epoch in range(total_epochs):
    model.train()
    running_loss = 0.0
    for i, ((inputs1, inputs2), labels) in enumerate(train_loader):
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

        inputs1 = inputs1.unsqueeze(1)  # Add channel dimension
        inputs2 = inputs2.unsqueeze(-1).permute(0, 2, 1)  # Adjust shape (batch_size, seq_len, input_size)

        optimizer.zero_grad()

        class_output, src_features = model(inputs1, inputs2)
        class_loss = criterion_class(class_output, labels)

        # Use source domain features to compute MMD loss
        src_features = src_features.detach()  # Only need to compute MMD loss
        tgt_features = src_features.clone().detach()  # Target domain features (placeholder example)
        mmd_loss = model.mmd_loss(src_features, tgt_features)

        total_loss = class_loss + mmd_loss
        total_loss.backward()
        optimizer.step()

        running_loss += class_loss.item()

    # print(f"Epoch [{epoch + 1}/{total_epochs}], Classification Loss: {running_loss / len(train_loader)}")

# Load target domain data
root_dir_target = 'gob_zone'
target_dataset = TargetCSVDataset(root_dir_target)
target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)

# Fine-tune model on target domain data
for epoch in range(total_epochs):
    model.train()
    running_loss = 0.0
    for i, ((inputs1, inputs2), labels) in enumerate(target_loader):
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

        inputs1 = inputs1.unsqueeze(1)  # Add channel dimension
        inputs2 = inputs2.unsqueeze(-1).permute(0, 2, 1)  # Adjust shape (batch_size, seq_len, input_size)

        optimizer.zero_grad()

        class_output, tgt_features = model(inputs1, inputs2)
        class_loss = criterion_class(class_output, labels)

        # Compute MMD loss
        src_features = torch.zeros_like(tgt_features)  # Source domain features' mean for MMD loss
        mmd_loss = model.mmd_loss(src_features, tgt_features)

        total_loss = class_loss + mmd_loss
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
    with open('DDC_01.txt', 'a') as f:
        f.write("第{}轮训练的准确率：{}\n".format(epoch + 1, str(test_accuracy)[0:5]+'%'))

    # 将模型设置回训练模式
    model.train()
    # Test the model's performance
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for (inputs1, inputs2), labels in target_loader:
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
        inputs1 = inputs1.unsqueeze(1)  # Add channel dimension
        inputs2 = inputs2.unsqueeze(-1).permute(0, 2, 1)  # Adjust shape (batch_size, seq_len, input_size)
        class_output, _ = model(inputs1, inputs2)
        _, predicted = torch.max(class_output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Model accuracy on target domain test data: {(correct / total) * 100:.2f}%")

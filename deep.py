import torch.utils.data
# from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
## 混合精度计算所需的库
from torch.cuda.amp import autocast, GradScaler
## 计时所需的库
import time
from datetime import timedelta

## 定义训练设备
device_0 = torch.device("cpu")
device_1 = torch.device("cuda:0")

######################## construct CBAMResNet  ###########################
# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes//ratio, kernel_size=1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_channels=in_planes//ratio, out_channels=in_planes, kernel_size=1, bias=False)  # 第二个卷积层，升维

        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重

# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.tong = ChannelAttention(in_planes, ratio)  # 通道注意力实例
        self.kong = SpatialAttention(kernel_size)       # 空间注意力实例

    def forward(self, x):
        out = x * self.tong(x)  # 使用通道注意力加权输入特征图
        out = out * self.kong(out)  # 使用空间注意力进一步加权特征图
        return out  # 返回最终的特征图

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Resblock, self).__init__()

        # 通常卷积层后接有bn层的话，nn.Conv2d()中的bias设置为关闭
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        ## CBAM注意力
        self.cbam = CBAM(out_channels, ratio=16)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                                          nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_channels)
                                        )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        cbam_out = self.cbam(out)

        return cbam_out

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()

        # 初始化操作，经过7*7卷积和池化，降采样
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 进入残差块
        self.layer1 = self.new_make_layer(in_channels=64, out_channels=64, blocks=2, stride=1)
        self.layer2 = self.new_make_layer(in_channels=64, out_channels=128, blocks=2, stride=2)
        self.layer3 = self.new_make_layer(in_channels=128, out_channels=256, blocks=2, stride=2)
        self.layer4 = self.new_make_layer(in_channels=256, out_channels=512, blocks=2, stride=2)

        # 平均池化，降采样
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接，分类
        self.fc = nn.Linear(512, num_classes)

    def new_make_layer(self, in_channels, out_channels, blocks, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layer = []
        for stride in strides:
            layer.append(Resblock(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

class CBAMResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(CBAMResNet18, self).__init__()
        self.cbamresnet18 = ResNet18(num_classes)
        self.cbamresnet18.fc = torch.nn.Sequential(  ## 修改输出层的结构
                                nn.Linear(512, 100),
                                nn.Dropout(0.5),
                                nn.LeakyReLU(0.1),
                                nn.Linear(100, 10),  ## 输出层
                                nn.LeakyReLU(0.1),
                                nn.Linear(10, num_classes)
                                                )

    def forward(self, x):
        out = self.cbamresnet18(x)
        return out
####################  CBAMResNet constructed  ############################

# -------------------------------------------------------------------------------------

######################### construct Deit-S  ############################
class DeitS(nn.Module):
    def __init__(self, num_classes=1000):
        super(DeitS, self).__init__()
        self.DeitS = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=0)
        self.DeitS.head = torch.nn.Sequential(  ## 修改输出层的结构
                            nn.Linear(384, 100),
                            nn.Dropout(0.5),
                            nn.LeakyReLU(0.1),
                            nn.Linear(100, 10),  ## 输出层
                            nn.LeakyReLU(0.1),
                            nn.Linear(10, num_classes)
                                             )

    def forward(self, x):
        out = self.DeitS(x)

        return out
#######################  Deit-S constructed  ###########################


class LoadCBAMResNet18Weight(nn.Module):
    def __init__(self, num_classes=1000):
        super(LoadCBAMResNet18Weight, self).__init__()

        self.SeResNet18_weighted_0 = CBAMResNet18(num_classes)
        self.SeResNet18_weighted = self.load_weight()
        self.SeResNet18_weighted.to(device_1)

    def load_weight(self):
        ## se-resnet18的权重
        seresnet18_now_weights_dict = self.SeResNet18_weighted_0.state_dict()
        ## resnet18官方的预训练权重
        # resnet18_official_weights_path = 'C:\\Users\\39106\\.cache\\torch\\hub\\checkpoints\\resnet18-f37072fd.pth'
        resnet18_official_weights_path = 'pretrainedmodel/resnet18-f37072fd.pth'
        resnet18_pre_weights = torch.load(resnet18_official_weights_path, map_location=device_1)
        for k in resnet18_pre_weights.keys():
            if k in seresnet18_now_weights_dict.keys() and not k.startswith('fc'):
                seresnet18_now_weights_dict[k] = resnet18_pre_weights[k]
        self.SeResNet18_weighted_0.load_state_dict(seresnet18_now_weights_dict)
        return self.SeResNet18_weighted_0

    def forward(self, x):
        out = self.SeResNet18_weighted(x)
        return out


class LoadDeitSWeight(nn.Module):
    def __init__(self, num_classes=1000):
        super(LoadDeitSWeight, self).__init__()

        self.DeitS_weighted_0 = DeitS(num_classes)
        self.DeitS_weighted = self.load_weight()
        self.DeitS_weighted.to(device_1)

    def load_weight(self):
        ## 获取模型被修改后的权重参数
        deits_now_weights_dict = self.DeitS_weighted_0.state_dict()
        ## 加载官方Deit-S预训练权重
        # DeitS_official_weights_path = 'C:\\Users\\39106\\.cache\\torch\\hub\\checkpoints\\deit_small_patch16_224-cd65a155.pth'
        DeitS_official_weights_path = 'pretrainedmodel/deit_small_patch16_224-cd65a155.pth'
        deits_pre_weights = torch.load(DeitS_official_weights_path, map_location=device_1)
        ## 判断预训练模型中网络的模块是否修改后的网络中也存在，并且shape相同，如果相同则取出
        deits_pretrained_dict = {k: v for k, v in deits_pre_weights.items() if k in deits_now_weights_dict and (v.shape == deits_now_weights_dict[k].shape)}
        ## 更新修改之后的 deits_now_weights_dict, 并加载到模型中
        deits_now_weights_dict.update(deits_pretrained_dict)
        self.DeitS_weighted_0.load_state_dict(deits_now_weights_dict, strict=False)

        return self.DeitS_weighted_0

    def forward(self, x):
        out = self.DeitS_weighted(x)
        return out


class CompleteCBAMResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(CompleteCBAMResNet18, self).__init__()
        self.complete_CBAMResNet18 = LoadCBAMResNet18Weight(num_classes)

    def forward(self, x):
        out = self.complete_CBAMResNet18(x)
        return out

class CompleteDeitS(nn.Module):
    def __init__(self, num_classes=1000):
        super(CompleteDeitS, self).__init__()
        self.complete_DeitS = LoadDeitSWeight(num_classes)

    def forward(self, x):
        out = self.complete_DeitS(x)
        return out


class Combine(nn.Module):
    def __init__(self, combine_size=100, num_classes=1000):
        super(Combine, self).__init__()
        self.cbamrn = CompleteCBAMResNet18(combine_size)
        self.deit = CompleteDeitS(combine_size)
        self.cla = nn.Sequential(
                            nn.Linear(in_features=2 * combine_size, out_features=combine_size),
                            nn.Dropout(0.3),
                            nn.LeakyReLU(0.1),
                            nn.Linear(in_features=combine_size, out_features=num_classes)
        )

    def forward(self, x):
        out1 = self.cbamrn(x)
        out2 = self.deit(x)
        out = torch.cat(tensors=(out1, out2), dim=1)
        out = self.cla(out)

        return out

#######################  final Model constructed  ###########################


pre_precess = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  ## 归一化
                                 ])

## 指定训练集与测试集
train_set = torchvision.datasets.ImageFolder(root='dataset2/train', transform=pre_precess)
val_set = torchvision.datasets.ImageFolder(root='dataset2/val', transform=pre_precess)

train_data_size = len(train_set)
val_data_size = len(val_set)
print('train_data_size:', train_data_size)
print('val_data_size:', val_data_size)

## 把数据集加载到dataloader
batch_size = 16
num_workers = 0  ## Windows系统不支持多进程，所以num_workers设为0
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

## 定义类别数量、名称
classes = ('class1', 'class2', 'class3')
num_classes = 6

## 实例化模型
print("实例化模型")
final_model = Combine(100, num_classes)
final_model.to(device_1)
# print(final_model)

## 设置超参数
learning_rate = 1e-4
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(device_1)

## 优化器
optimizer = torch.optim.AdamW(final_model.parameters(), lr=learning_rate)

epochs = 100
total_train_step = 0  # 训练次数计数器
total_val_step = 0   # 测试次数计数器

## 添加Tensorboard
# writer = SummaryWriter(log_dir="log/cbam_deit/0613_autodl")

## 实例化GradScale，用于放缩
scaler = GradScaler()

## 训练and测试
for i in range(epochs):
    ## ----------------------------------------------------------------
    ## 训练开始
    ## 计时开始，统计每Epoch训练用时
    start_time = time.perf_counter()
    print("----------第{}轮训练开始----------".format(i+1))

    total_train_accuracy = 0
    total_train_loss = 0
    final_model.train()
    for data in train_loader:
        input_imgs, targets = data            # 取数据、标签
        input_imgs = input_imgs.to(device_1)  # 送入GPU
        targets = targets.to(device_1)        # 送入GPU

        ## 混合精度计算
        with autocast():
            output_imgs = final_model(input_imgs)
            train_loss = loss_fn(output_imgs, targets)

        train_accuracy = (output_imgs.argmax(1) == targets).sum()
        total_train_accuracy = total_train_accuracy + train_accuracy
        total_train_loss = total_train_loss + train_loss

        ## 开始优化
        optimizer.zero_grad()   # 梯度清零
        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_train_step = total_train_step + 1  # 计数器 + 1

        ## 打印信息。逢100次训练，打印一次
        if total_train_step % 60 == 0:
            print("训练次数：{}，train_loss：{}".format(total_train_step, train_loss))
            # writer.add_scalar("train_loss", train_loss, total_train_step)

    ## 计时结束
    end_time = time.perf_counter()
    cost_time = timedelta(seconds=(end_time - start_time))

    ## 训练正确率
    print("整体训练集上的正确率：{}".format(total_train_accuracy/train_data_size))
    # writer.add_scalar("total_train_loss", total_train_loss, total_train_step)
    # writer.add_scalar("train_accuracy", total_train_accuracy/train_data_size, total_train_step)
    print('整体训练集上的train_loss:{}'.format(total_train_loss))
    print('第{}轮训练用时：{}'.format(i+1, cost_time))

    # ------------------------------------------------------------------------------
    ## 在一轮训练结束后 , 进行测试
    total_val_loss = 0
    total_accuracy = 0
    final_model.eval()
    with torch.no_grad():
        for data in val_loader:
            input_imgs, targets = data
            input_imgs = input_imgs.to(device_1)
            targets = targets.to(device_1)
            output_imgs = final_model(input_imgs)
            # print(output_imgs)
            # print('-'*16)
            # print(targets)
            val_loss = loss_fn(output_imgs, targets)
            total_val_loss = total_val_loss + val_loss
            accuracy = (output_imgs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

        print("整体测试集上的val_loss:{}".format(total_val_loss))
        print("整体测试集上的正确率：{}".format(total_accuracy / val_data_size))
        total_val_step = total_val_step + 1
        # writer.add_scalar("total_val_loss", total_val_loss, total_val_step)
        # writer.add_scalar("val_accuracy", total_accuracy / val_data_size, total_val_step)

    #
    if total_accuracy/val_data_size > 0.955:
        torch.save(final_model, "lianhe_Epoch-{}.pth".format(i+1))
        print("lianhe_Epoch-{}.pth  has been saved!".format(i+1))

# writer.close()
# 打开tensorboard命令：tensorboard --logdir=log/train --port=6061
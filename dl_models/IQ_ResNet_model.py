import torch
import torch.nn as nn
import os


class ResidualBlock(nn.Module):
    """
    对应图 3 中的残差块结构：
    包含两个 1x3 卷积，第一层步长为 (1, 2) 用于下采样，第二层步长为 (1, 1)。
    Shortcut(旁路) 经过 1x1 卷积，步长 (1, 2) 以对齐维度。
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        # 1. 主路径：第一个 1x3 卷积，步长 (1, 2) 减半时间维度
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3),
                               stride=(1, 2), padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2. 主路径：第二个 1x3 卷积，步长 (1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3),
                               stride=(1, 1), padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 3. 旁路 Shortcut：图 3 中标注为 convXX, (1, 2)
        # 根据论文 Section III-C "passes the input x through a 1x1 convolution layer"
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1),
                      stride=(1, 2), bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # F(x) + x
        out = self.relu(out)
        return out


class IQ_ResNet(nn.Module):
    def __init__(self, num_classes=181):
        super(IQ_ResNet, self).__init__()

        # 1. 初始特征提取层: 16x5 conv64, stride (16, 1)
        # 输入 shape: (Batch, 1, 16, T)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(16, 5), stride=(16, 1), padding=(0, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 2. 最大池化层: 1x3 maxpool, stride (1, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        # 3. 四个残差块: 输出通道分别为 64, 128, 256, 512
        self.layer1 = ResidualBlock(64, 64)  # 橘色块
        self.layer2 = ResidualBlock(64, 128)  # 蓝色块
        self.layer3 = ResidualBlock(128, 256)  # 绿色块
        self.layer4 = ResidualBlock(256, 512)  # 浅蓝色块

        # 4. 全局平均池化 (avgpool)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 5. 全连接层 (fc)
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        # 确保输入是 4D tensor: (Batch, Channels, M*2, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)  # 输出 Logits

        return out
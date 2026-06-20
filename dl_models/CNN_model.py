import torch
import torch.nn as nn

class CNN_Classify(nn.Module):
    def __init__(self, num_classes=181):
        super(CNN_Classify, self).__init__()
        #  2 通道输入 (Real, Imag)
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class CNN_Regression(nn.Module):
    def __init__(self, out_dim=1):
        super(CNN_Regression, self).__init__()
        self.features = CNN_Classify().features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, out_dim) # 👇 把原本写死的 1 改成 out_dim
        )

    def forward(self, x):
        return self.classifier(self.features(x))
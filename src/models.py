import torch.nn as nn
import torch
import torchvision import models


class Resnet18_Pretrained_Backbone(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        resnet = models.resnet18(pretrained=True)

        # Remove the last fc layer of the pretrained network.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze backbone weights.
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Implement the fully connected layers for classification and regression.
        self.classification = nn.Sequential(nn.Flatten(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_classes), nn.ReLU())

    def forward(self, x):
        x = self.backbone(x)
        x = self.classification(x)
        return x

class CNN_16_32_FC_128(nn.Module):
    def __init__(self, num_classes=2, num_channels=3, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_32_64_FC_128(nn.Module):
    def __init__(self, num_classes=2, num_channels=3, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_64_128_FC_256(nn.Module):
    def __init__(self, num_classes=2, num_channels=3, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(64, 128, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_16_32_FC_128_64(nn.Module):
    def __init__(self, num_classes=2, num_channels=3, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, stride, padding)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

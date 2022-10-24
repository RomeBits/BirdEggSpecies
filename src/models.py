import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, num_classes=2, num_channels=3, kernel_size=3, stride=1, padding=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, stride, padding)
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

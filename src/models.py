import torch.nn as nn
import torch
from torchvision.models import resnet18, resnet50
from torchvision.models import vit_b_16
from torchvision.models import resnext101_32x8d


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        resnet = resnet18(weights=None)

        # Remove the last fc layer of the pretrained network.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(512)
        self.classification = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_classes), nn.ReLU())

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.classification(x)
        return x


class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        resnet = resnet50(weights=None)

        # Remove the last fc layer of the pretrained network.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.classification = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, num_classes), nn.ReLU())

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x


class CNN_16_32_FC_128(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Sequential(nn.Linear(32 * 40 * 40, 128), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, num_classes), nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class CNN_32_64_FC_128(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Sequential(nn.Linear(64 * 40 * 40, 128), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, num_classes), nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = vit_b_16(weights=None)
        self.classification = nn.Sequential(nn.Linear(1000, 512), nn.ReLU(), nn.Linear(512, num_classes), nn.ReLU())

    def forward(self, x):
        # resize to 224x224
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.backbone(x)
        x = self.classification(x)
        return x


class Resnext101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        resnet = resnext101_32x8d(weights=None)

        # Remove the last fc layer of the pretrained network.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.classification = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, num_classes), nn.ReLU())

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x
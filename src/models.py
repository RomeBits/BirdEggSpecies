from torchvision import models
import torch.nn as nn
import torch

class ResNet18Embedding(models.ResNet):
    def __init__(self, embedding_dim=128):
        super().__init__(models.resnet.BasicBlock, [2, 2, 2, 2])
        self.fc = nn.Linear(512, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CNNClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_classes):
        super().__init__()
        self.embedding = ResNet18Embedding(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.embedding.forward(x)
        x = self.classifier(x)
        return x

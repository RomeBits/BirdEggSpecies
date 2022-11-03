from ..models import CNN_16_32_FC_128, CNN_32_64_FC_128, CNN_64_128_FC_256, CNN_16_32_FC_128_64
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, TensorDataset

import torch
import numpy as np


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train_classifier(model, train_loader, valid_loader, device, epochs=10, lr=0.001):
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        acc = test(model, valid_loader, device)
        print(f"Epoch {epoch + 1}/{epochs} - Test accuracy: {acc}")
    return model


if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="CNN_16_32_FC_128")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="models")
    args = parser.parse_args()


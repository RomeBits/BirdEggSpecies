from models import CNN_16_32_FC_128, CNN_32_64_FC_128, CNN_64_128_FC_256, CNN_16_32_FC_128_64
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import numpy as np


class TransformedTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


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
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, len(train_loader), loss.item()))
        acc = test(model, valid_loader, device)
        print(f"Epoch {epoch + 1}/{epochs} - Validation accuracy: {acc}")
    return model


if __name__ == "__main__":
    import argparse
    import os
    import sys
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="CNN_16_32_FC_128")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")

    # load label encoder
    with open(os.path.join(args.data_dir, "objects/le.pkl"), "rb") as f:
        train_label_encoder = pickle.load(f)

    with open(os.path.join(args.data_dir, "objects/le.pkl"), "rb") as f:
        test_label_encoder = pickle.load(f)

    # Load data
    data = np.load(os.path.join(args.data_dir, "train_dataset.npz"), allow_pickle=True)
    X_train = np.transpose(data["X"], (0, 3, 1, 2))
    y_train = train_label_encoder.transform(data["Y"])

    data = np.load(os.path.join(args.data_dir, "test_dataset.npz"), allow_pickle=True)
    X_valid = np.transpose(data["X"], (0, 3, 1, 2))
    y_valid = test_label_encoder.transform(data["Y"])
    # print("Splitting data...")

    # Split data
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # create transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((64, 64))
    ])

    # Create data loaders
    train_dataset = TransformedTensorDataset(
        tensors=(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
        transform=transform
    )
    valid_dataset = TransformedTensorDataset(
        tensors=(torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid).long()),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    print("Training model...")

    # Create model
    model = eval(args.model)(num_classes=len(train_label_encoder.classes_))

    # Train model
    model = train_classifier(model, train_loader, valid_loader, device, epochs=args.epochs, lr=args.lr)

    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "{}_ben.pth".format(args.model)))

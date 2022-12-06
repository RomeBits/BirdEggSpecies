from models import *
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

import torch
import numpy as np


def plot_loss_accuracy(train_loss, test_loss, train_acc, test_acc):
    # convert tensors to numpy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Valid')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train')
    plt.plot(test_acc, label='Valid')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

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


def test(model, test_loader, device, output_dir=None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # plot failed images
            if output_dir:
                for i, label in enumerate(labels):
                    if predicted[i] != label:
                        img = images[i].cpu().numpy()
                        img = np.transpose(img, (1,2,0))
                        plt.imsave(f"{output_dir}/{label}_{predicted[i]}.png", img)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += criterion(outputs, labels).item()
    return correct / total, total_loss / len(test_loader)


def train_classifier(model, train_loader, valid_loader, device, epochs=10, lr=0.001):
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    accuracies = []
    valid_acc = []
    valid_loss = []
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total += len(labels)
            total_loss += loss.item()
            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, len(train_loader), loss.item()))
        losses.append(total_loss / len(train_loader))
        accuracies.append(correct / total)
        acc, val_loss = test(model, valid_loader, device)
        valid_acc.append(acc)
        valid_loss.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Validation accuracy: {acc}, Training accuracy: {correct / float(len(train_loader.dataset))}")
    plot_loss_accuracy(losses, valid_loss, accuracies, valid_acc)
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
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    print("Loading data...")

    # load label encoder
    with open(os.path.join(args.data_dir, "objects/le.pkl"), "rb") as f:
        train_label_encoder = pickle.load(f)

    with open(os.path.join(args.data_dir, "objects/le.pkl"), "rb") as f:
        test_label_encoder = pickle.load(f)

    # Load data
    data = np.load(os.path.join(args.data_dir, "train_dataset.npz"), allow_pickle=True)
    X_raw = np.transpose(data["X"], (0, 3, 1, 2))
    y_raw = train_label_encoder.transform(data["Y"])

    data = np.load(os.path.join(args.data_dir, "test_dataset.npz"), allow_pickle=True)
    X_test = np.transpose(data["X"], (0, 3, 1, 2))
    y_test = test_label_encoder.transform(data["Y"])

    X_raw = np.flip(X_raw, axis=1).copy()
    X_test = np.flip(X_test, axis=1).copy()

    print("Splitting data...")

    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    # create transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160,160)),
        transforms.ToTensor()
    ])

    # Create data loaders
    train_dataset = TransformedTensorDataset(
        tensors=(torch.from_numpy(X_train), torch.from_numpy(y_train).long()),
        transform=transform
    )
    valid_dataset = TransformedTensorDataset(
        tensors=(torch.from_numpy(X_valid), torch.from_numpy(y_valid).long()),
        transform=transform
    )
    test_dataset = TransformedTensorDataset(
        tensors=(torch.from_numpy(X_test), torch.from_numpy(y_test).long()),
        transform=transform
    )

    print("Input image size:", X_train.shape[1:])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Training model...")

    # Create model
    model = eval(args.model)(num_classes=len(train_label_encoder.classes_))

    # load model
    if args.pretrained:
        model.load_state_dict(torch.load(args.output_dir + "/{}.pth".format(args.model)))

    # Train model
    model = train_classifier(model, train_loader, valid_loader, device, epochs=args.epochs, lr=args.lr)

    # Test model
    acc = test(model, test_loader, device, output_dir="../images/failed")
    print(f"Test accuracy: {acc[0]}")

    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "{}.pth".format(args.model)))

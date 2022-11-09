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


if __name__ == "__main__":
    import argparse
    import os
    import sys
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="CNN_16_32_FC_128")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--model_dir", type=str, default="../models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")

    # load label encoder
    with open(os.path.join(args.data_dir, "objects/le.pkl"), "rb") as f:
        train_label_encoder = pickle.load(f)

    with open(os.path.join(args.data_dir, "objects/le.pkl"), "rb") as f:
        test_label_encoder = pickle.load(f)

    data = np.load(os.path.join(args.data_dir, "test_dataset.npz"), allow_pickle=True)
    X_test = np.transpose(data["X"], (0, 3, 1, 2))
    y_test = test_label_encoder.transform(data["Y"])

    # create transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((64, 64))
    ])

    # Create data loader
    test_dataset = TransformedTensorDataset(
        tensors=(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()),
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Testing model...")

    # Create model
    model = eval(args.model)(num_classes=len(train_label_encoder.classes_))

    # Load model
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "{}.pth".format(args.model))))

    # Move model to device
    model.to(device)

    # Test model
    accuracy = test(model, test_loader, device)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

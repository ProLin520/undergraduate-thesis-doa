import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dl_models.IQ_ResNet_model import IQ_ResNet


current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(current_dir))


def load_dataset(base_path):
    train_data = np.load(os.path.join(base_path, "Train", "train_data.npy"))
    train_labels = np.load(os.path.join(base_path, "Train", "train_labels.npy"))
    val_data = np.load(os.path.join(base_path, "Val", "val_data.npy"))
    val_labels = np.load(os.path.join(base_path, "Val", "val_labels.npy"))

    x_train = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(np.argmax(train_labels, axis=1), dtype=torch.long)
    x_val = torch.tensor(val_data, dtype=torch.float32)
    y_val = torch.tensor(np.argmax(val_labels, axis=1), dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False)
    return train_loader, val_loader


def train_and_infer_single_source(train_loader, val_loader, rho, device="cuda"):
    model = IQ_ResNet(num_classes=181).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 30
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Training IQ-ResNet with rho={rho} on {device}")

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    save_path = os.path.join(root_path, "result", "IQ_ResNet", "SingleSource",
                             f"IQ_ResNet_SingleSource_rho{rho}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_path)
    print(f"Saved best IQ-ResNet to: {save_path}")


if __name__ == "__main__":
    rho = 0.0
    data_base = os.path.join(root_path, "data", "IQ_Data", "Single_Source", f"Single_Source_Rho{rho}")

    if not os.path.exists(data_base):
        print(f"Dataset path not found: {data_base}")
    else:
        train_loader, val_loader = load_dataset(data_base)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_and_infer_single_source(train_loader, val_loader, rho=rho, device=device)

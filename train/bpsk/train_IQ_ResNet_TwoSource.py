import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dl_models.IQ_ResNet_model import IQ_ResNet


current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(current_dir))


def load_two_source_dataset(base_path):
    snrs = np.arange(-25, 26, 5)

    def load_split(split_name):
        data_list, label_list = [], []
        print(f"Loading {split_name} data...")
        for snr in tqdm(snrs):
            data_path = os.path.join(base_path, split_name, f"{split_name.lower()}_data_snr{snr}.npy")
            label_path = os.path.join(base_path, split_name, f"{split_name.lower()}_labels_snr{snr}.npy")
            data_list.append(np.load(data_path))
            label_list.append(np.load(label_path))

        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        return torch.from_numpy(data).float().unsqueeze(1), torch.from_numpy(labels).float()

    x_train, y_train = load_split("Train")
    x_val, y_val = load_split("Val")

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False, pin_memory=True)
    return train_loader, val_loader


def train_and_infer_two_source(train_loader, val_loader, rho, device="cuda"):
    model = IQ_ResNet(num_classes=181).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    num_epochs = 60
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Training two-source IQ-ResNet with rho={rho} on {device}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted_indices = torch.topk(outputs, 2, dim=1)
                predicted_multi_hot = torch.zeros_like(labels).scatter_(1, predicted_indices, 1)
                total += labels.size(0)
                correct += (predicted_multi_hot == labels).all(dim=1).sum().item()

        val_acc = correct / total
        epoch_loss = running_loss / len(train_loader.dataset)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Exact Match Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    save_path = os.path.join(root_path, "result", "IQ_ResNet", "TwoSource",
                             f"IQ_ResNet_TwoSource_rho{rho}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_path)
    print(f"Saved best IQ-ResNet to: {save_path}")


if __name__ == "__main__":
    rho = 1.0
    data_base = os.path.join(root_path, "data", "IQ_Data", "Two_Source", f"Two_Source_Rho{rho}")
    if not os.path.exists(data_base):
        print(f"Dataset path not found: {data_base}")
    else:
        train_loader, val_loader = load_two_source_dataset(data_base)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_and_infer_two_source(train_loader, val_loader, rho=rho, device=device)

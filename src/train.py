import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_dataloaders
from model import get_model
from utils import load_config, save_model

def train():
    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        image_size=config["image_size"]
    )

    model = get_model(
        model_name=config["model_name"],
        num_classes=config["num_classes"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {avg_loss:.4f}")

    save_model(model, f"models/{config['model_name']}_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()
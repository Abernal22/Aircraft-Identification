import torch
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import get_dataloaders
from model import get_model
from utils import load_config

def evaluate():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        image_size=config["image_size"]
    )

    model = get_model(
        model_name=config["model_name"],
        num_classes=config["num_classes"]
    ).to(device)

    model.load_state_dict(torch.load(f"models/{config['model_name']}_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()
from torchvision import datasets
from torch.utils.data import DataLoader
from augmentations import get_train_transforms, get_eval_transforms

def get_dataloaders(data_dir, batch_size=32, image_size=224):
    train_transform = get_train_transforms(image_size)
    eval_transform = get_eval_transforms(image_size)

    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(f"{data_dir}/test", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loadert_dataset
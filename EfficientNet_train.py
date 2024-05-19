import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import argparse
import timm
import torch
from torch import nn, optim
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

class DataLoaderWrapper:
    def __init__(self, data_dir, batch_size=64, image_size=(240, 240)):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),  # EfficientNet-B1 принимает изображения 240x240
            transforms.ToTensor(),
        ])

        self.train_loader = None
        self.valid_loader = None
        self.num_classes = None

    def load_valid_images(self, root):
        valid_samples = []
        dataset = datasets.ImageFolder(root=root, transform=self.transform)
        for sample in dataset.samples:
            try:
                with Image.open(sample[0]) as img:
                    img.verify()  # Проверка изображения
                    img.close()
                valid_samples.append(sample)
            except (IOError, OSError, ValueError) as e:
                print(f"Ошибка при загрузке изображения {sample[0]}: {e}")
        dataset.samples = valid_samples
        return dataset

    def setup_loaders(self):
        train_dir = os.path.join(self.data_dir, 'train')
        valid_dir = os.path.join(self.data_dir, 'valid')

        train_dataset = self.load_valid_images(train_dir)
        valid_dataset = self.load_valid_images(valid_dir)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.num_classes = len(train_dataset.classes)


    def get_train_loader(self):
        if self.train_loader is None:
            self.setup_loaders()
        return self.train_loader

    def get_valid_loader(self):
        if self.valid_loader is None:
            self.setup_loaders()
        return self.valid_loader

    def get_count_classes(self):
        return self.num_classes


def train_and_validate(data_dir, save_path, num_epochs, device):
    data_loader_wrapper = DataLoaderWrapper(data_dir)

    train_loader = data_loader_wrapper.get_train_loader()
    val_loader = data_loader_wrapper.get_valid_loader()

    model_name = 'efficientnet_b1'
    model = timm.create_model(model_name, pretrained=True, num_classes=data_loader_wrapper.get_count_classes())

    # Определение устройства
    model.to(device)

    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    print('Start training')
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

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}')

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}, Val Accuracy: {val_accuracy}'")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} with validation loss: {best_val_loss}")

    print('End training')
    return model


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

    return epoch_loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate model.")
    parser.add_argument("data_dir", type=str, help="Path to the input data directory.")
    parser.add_argument("num_epochs", type=int, help="Number of epochs for training.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Device to run the model on.")

    args = parser.parse_args()

    # Обучение и валидация модели и сохранение
    trained_model = train_and_validate(args.data_dir, save_path="best_model_efficientnet.pth",
                                       num_epochs=args.num_epochs, device=args.device)

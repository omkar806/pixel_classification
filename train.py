import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

def setup_device():
    """
    Set up the appropriate device for training on Mac.
    Returns the device and prints its properties.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
    return device

class GeoTreeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_map = {'GROUND': 0, 'LIVE': 1, 'DEAD': 2, 'N/A': -1}
        self.data = self.data[self.data['Class label'] != 'N/A'].copy()
        self.data['label_num'] = self.data['Class label'].map(self.label_map)
        self.data['label_num'] = self.data['label_num'].fillna(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['Image path']
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx]['label_num'])  # Ensure it's an integer

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

class TreeClassifierCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(TreeClassifierCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        print(f"Feature map shape before flattening: {x.shape}")
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cpu'):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        history['train_loss'].append(running_loss / len(train_loader))
        history['train_acc'].append(100 * correct / total)
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(100 * correct / total)
        print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {history["train_loss"][-1]:.4f}, Train Acc: {history["train_acc"][-1]:.2f}%, Val Loss: {history["val_loss"][-1]:.4f}, Val Acc: {history["val_acc"][-1]:.2f}%')
        torch.save(model.state_dict(), 'tree_classifier.pth')
        print("Model saved successfully as 'tree_classifier.pth'.")
    return history

def main():
    device = setup_device()
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = GeoTreeDataset('/Users/omkarmalpure/Documents/pixel_classification/pixel_dataset - labeled.csv', transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    model = TreeClassifierCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device=device)

if __name__ == "__main__":
    main()

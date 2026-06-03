import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device:{device}")

train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(root="data", train=True,  download=True, transform=train_transform)
test_dataset  = datasets.MNIST(root="data", train=False, download=True, transform=test_transform)

print(f"Train Samples:{len(train_dataset)},Test samples:{len(test_dataset)}")


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == '__main__':
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = MnistCNN().to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters:{total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    def run_epoch(loader, training=True):
        model.train() if training else model.eval()
        total_loss, correct = 0.0, 0
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                if training:
                    optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                if training:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * len(labels)  # ← was + (addition), must be * (multiplication)
                correct += (outputs.argmax(1) == labels).sum().item()
        n = len(loader.dataset)
        return total_loss / n, correct / n

    EPOCHS = 20
    PATIENCE = 5
    best_val_acc, patience_counter = 0.0, 0

    os.makedirs("models", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(train_loader, training=True)
        vl_loss, vl_acc = run_epoch(test_loader,  training=False)
        scheduler.step(vl_loss)
        print(f"Epoch {epoch:02d}/{EPOCHS} "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_counter = 0
            torch.save(model.state_dict(), "models/mnist_cnn.pth")
            print(f"  saved best model (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping after {epoch} epochs.")
                break

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print("Model saved to models/mnist_cnn.pth")
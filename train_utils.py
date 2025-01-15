# train_utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

def train_model(model, device, lr, batch_size, epochs, progress_callback=None):
    """
    Trains the model on MNIST using specified hyperparams.
    Optionally calls `progress_callback(epoch, total_epochs)` to update a GUI.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs+1):
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
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        if progress_callback:
            progress_callback(epoch, epochs)

    print("Training complete.")

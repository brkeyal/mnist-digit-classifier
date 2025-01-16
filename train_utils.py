# train_utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import psutil

def train_model(model, device, lr, batch_size, epochs, progress_callback=None):
    """
    Trains the model on MNIST using specified hyperparameters.
    Optionally calls `progress_callback(epoch, total_epochs)` to update a GUI.
    Measures training time and memory usage.
    Returns a dictionary with metrics.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    # Record the start time
    start_time = time.time()

    # Record initial memory usage
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 ** 2)  # in MB

    # Record initial GPU memory usage (if applicable)
    if device.type == 'cuda':
        gpu_mem_before = torch.cuda.memory_allocated(device) / (1024 ** 2)  # in MB
    else:
        gpu_mem_before = 0

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
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

    # Record the end time
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time for {epochs} epochs: {training_time:.2f} seconds")

    # Record final memory usage
    mem_after = process.memory_info().rss / (1024 ** 2)  # in MB
    mem_used = mem_after - mem_before
    print(f"CPU Memory Usage During Training: {mem_used:.2f} MB")

    # Record final GPU memory usage (if applicable)
    if device.type == 'cuda':
        gpu_mem_after = torch.cuda.memory_allocated(device) / (1024 ** 2)  # in MB
        gpu_mem_used = gpu_mem_after - gpu_mem_before
        print(f"GPU Memory Usage During Training: {gpu_mem_used:.2f} MB")
    else:
        gpu_mem_used = 0
        print("GPU not available; skipping GPU memory usage tracking.")

    # Return the metrics as a dictionary
    return {
        'training_time': training_time,
        'cpu_memory_usage': mem_used,
        'gpu_memory_usage': gpu_mem_used
    }

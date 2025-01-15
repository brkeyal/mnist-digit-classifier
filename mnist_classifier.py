import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# Hyperparameters
BATCH_SIZE = 64      # The number of training samples processed in one forward/backward pass
EPOCHS = 5           # How many times the entire training dataset is passed through the network
LEARNING_RATE = 0.001  # Controls the step size at each update (how fast the model learns)

###########################################
# Data Loading & Transformations ##########
##########################################


# Transforms: convert images to tensors and normalize
# MNIST images are grayscale, mean=0.1307, std=0.3081 are typical stats for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and create training & test datasets
train_dataset = datasets.MNIST(
    root='data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root='data',
    train=False,
    transform=transform,
    download=True
)

# Create DataLoaders for training and testing
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

###########################################
#  Defining the Model ##########
##########################################
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)   # first hidden layer
        self.fc2 = nn.Linear(128, 64)     # second hidden layer
        self.fc3 = nn.Linear(64, 10)      # output layer (10 classes for digits 0–9)

    def forward(self, x):
        # Flatten the image from [batch, 1, 28, 28] to [batch, 28*28]
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # final output (logits)
        return x

model = SimpleNN()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model_path = 'mnist_model.pth'

if os.path.exists(model_path):
    # If there's a saved model, load it
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Loaded existing model from {model_path}.\n")
else:
    # Otherwise, train the model from scratch
    print("No saved model found, starting training...\n")

    ###########################################
    #  Define Loss Function & Optimizer ##########
    ##########################################
    criterion = nn.CrossEntropyLoss()             # Cross entropy is common for classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ###########################################
    #  Training Loop ##########
    ##########################################
    for epoch in range(EPOCHS):
        model.train()  # set model to training mode
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()           # reset gradients
            outputs = model(images)         # forward pass
            loss = criterion(outputs, labels)
            loss.backward()                 # backprop
            optimizer.step()                # update weights

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'mnist_model.pth')
    print(f"\nModel saved as {model_path}.\n")

###########################################
#  Testing the Model ##########
##########################################
model.eval()  # set model to evaluation mode
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

###########################################
# Classifying a Custom Image ##########
##########################################
from PIL import Image
import torchvision.transforms as T

def predict_image(model, image_path):
    # transform for a single image
    transform = T.Compose([
        T.Grayscale(),                   # ensure image is grayscale
        T.Resize((28, 28)),             # resize to 28x28
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])

    # open the image and transform
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0).to(device)  # add batch dimension

    # forward pass
    model.eval()
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# usage
# image_path = "my_digit.jpg"
# digit_prediction = predict_image(model, image_path)
# print("Predicted digit:", digit_prediction)


import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define transformations for data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load training and test datasets
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define the CNN model
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.cnn = Sequential(
            Conv2d(in_channels=1, out_channels=8, kernel_size=5),  # Initial configuration
            MaxPool2d(kernel_size=2, stride=2),
            # Add more convolutional layers here if desired
        )
        self.fc = Sequential(
            Flatten(),
            Linear(in_features=7 * 7 * 8, out_features=10),  # Output layer for 10 classes
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

# Initialize the model
model = FashionCNN()

# Define loss function (CrossEntropyLoss for multi-class classification)
criterion = nn.CrossEntropyLoss()

# Define an optimizer (here using Adam)
optimizer = torch.optim.Adam(model.parameters())

# Function to train the model
def train(epochs, model, train_loader, criterion, optimizer):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the weights

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Example usage
train(epochs=10, model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer)
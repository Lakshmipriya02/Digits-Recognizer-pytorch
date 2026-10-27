"""Train the notebook's model on MNIST and save the weights for the app.

Run once:  python train.py
Creates:   digits_classifier.pt
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import DigitsClassifier

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # must match preprocess.py
])

train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)

model = DigitsClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for images, labels in train_loader:
        loss = criterion(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item()
    # print(f"Epoch [{epoch + 1}/{EPOCHS}]  loss: {running / len(train_loader):.4f}")

model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
# print(f"Test accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "digits_classifier.pt")
print("Saved digits_classifier.pt")

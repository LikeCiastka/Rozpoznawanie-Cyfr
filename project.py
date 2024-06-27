import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import gradio as gr
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import matplotlib.pyplot as plt

global device

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'using {device}')

#torch.set_default_device(device)

# Załadowanie zbioru MNIST

transform = transforms.Compose([transforms.ToTensor()])
mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Definicja modelu CNN

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Funkcja do trenowania modelu
def train_model_cnn(num_epochs=10):
    # Załadowanie MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    #train_dataset.dataset.to(device)
    #val_dataset.dataset.to(device)
    
    # DataLoadery
    train_loader = DataLoader(train_dataset, batch_size=7500, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=7500, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Inicjalizacja modelu
    model = CNNModel()

    model.to(device=device)

    # Definicja funkcji straty i optymalizatora
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Walidacja modelu
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Ewaluacja modelu na zbiorze testowym
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    test_accuracy = 100 * correct / total
    print(f'Accuracy of the CNN model on the test set: {test_accuracy:.2f}%')

    # Zapis modelu
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    return model, test_accuracy

# Trenowanie modelu CNN
#if __name__ == "__main__":
#    train_model_cnn()

# Funkcja do załadowania modelu ResNet-18
def load_pretrained_resnet(num_classes):
    model = models.resnet18(pretrained=False)

    # Zmiana pierwszej warstwy Conv2d, aby obsługiwała obrazy o 1 kanale (skala szarości)
    # W oryginalnym ResNet-18 jest to warstwa Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Zamiana ostatniej warstwy Fully Connected
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

# Trenowanie modelu ResNet-18
def train_model_transfer_learning(model, train_loader, val_loader, test_loader, num_epochs=15, device=device):
    # Definicja funkcji straty i optymalizatora
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    val_losses = []

    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Walidacja modelu
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Ewaluacja modelu na zbiorze testowym
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    test_accuracy = 100 * correct / total
    print(f'Accuracy of the Transfer Learning model on the test set: {test_accuracy:.2f}%')

    # Zapis modelu
    torch.save(model.state_dict(), 'mnist_transfer_learning.pth')
    return model, test_accuracy

def classify_image(model, image):
    if image is None:
        return "no image"

    if isinstance(image, Image.Image):
        image = image.convert('L')  # Konwersja na skalę szarości
        image = image.resize((28, 28))  # Ustawienie rozmiaru
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image = transform(image)  # Normalizacja i konwersja do tensora
        image = image.unsqueeze(0)  # Dodanie wymiaru batcha
    else:
        return "invalid image"

    with torch.no_grad():
        model.eval()
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return str(predicted.item())

def load_mnist_data(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_size = int(0.7 * len(mnist_trainset))
    val_size = len(mnist_trainset) - train_size
    train_dataset, val_dataset = random_split(mnist_trainset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(mnist_testset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
    
# Załadowanie wytrenowanego modelu
model_path = 'mnist_transfer_learning.pth'
num_classes = 10  # MNIST ma 10 klas

# Definicja modelu
class ResNetMNIST(nn.Module):
    def __init__(self):
        super(ResNetMNIST, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)

# Inicjalizacja modelu i wczytanie wag
#model = ResNetMNIST()
#model.load_state_dict(torch.load(model_path))
#model.eval()

# Funkcja do klasyfikacji pojedynczego obrazu
def classify_image(image):
    if image is None:
        return "no image"

    if isinstance(image, Image.Image):
        image = image.convert('L')  # Konwersja na skalę szarości
        image = image.resize((28, 28))  # Ustawienie rozmiaru
        image = np.array(image).astype(np.float32) / 255.0  # Normalizacja
        image = torch.tensor(image, device=device).unsqueeze(0).unsqueeze(0)  # Konwersja na tensor
    else:
        return "invalid image"

    with torch.no_grad():
        outputs = model(image)
        sm = nn.Softmax(dim=1)
        outputs = sm(outputs)
        _, predicted = torch.max(outputs, 1)
        print(outputs)
        print(torch.max(outputs))
        return str(predicted.item())

# Interfejs Gradio
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Input Image"),  # Typ obrazu PIL
    outputs=gr.Label(num_top_classes=1),  # Wyjście jako etykieta
    live=True,
    #capture_session=True  # Zachowaj sesję interfejsu
)


global model


if __name__ == "__main__":
    # Załadowanie danych MNIST
    train_loader, val_loader, test_loader = load_mnist_data(batch_size=10000)

    # Inicjalizacja modelu ResNet-18 z transfer learningiem
    num_classes = 10  # MNIST ma 10 klas (cyfry od 0 do 9)
    model_transfer_learning = load_pretrained_resnet(num_classes)

    # Trenowanie modelu z transfer learningiem
    global model
    #model, test_acc = train_model_transfer_learning(model_transfer_learning, train_loader, val_loader, test_loader, device=device)
    
    model, test_acc = train_model_cnn(num_epochs=15)    
    # Uruchamianie interfejsu Gradio

    iface.launch(server_name="0.0.0.0", server_port=7860)

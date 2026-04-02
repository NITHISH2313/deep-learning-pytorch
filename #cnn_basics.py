#cnn_basics

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#==== LOAD REAL DATASET ========
# MNIST - handwritten digits 0-9
# 60,000 training images

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])

# Download training data
train_data = torchvision.datasets.MNIST(
    root = ',/data',
    train = True,
    download = True,
    transform = transform
)

test_data = torchvision.datasets.MNIST(
    root = ',/data',
    train = False,
    download = True,
    transform = transform

)

train_loader = torch.utils.data.DataLoader(train_data , batch_size = 64 , shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data , batch_size = 64 , shuffle = False)

print(f"training images :{len(train_data)}")
print(f"testing images : {len(test_data)}")
print(f"image shape : {train_data[0][0].shape}")

# ===== BUILD CNN ======

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1 , 16 , kernel_size = 3)
        self.conv2 = nn.Conv2d(16 , 32 , kernel_size = 3)
        self.pool = nn.MaxPool2d(2 ,2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(32 * 5 * 5 ,128)
        self.fc2 = nn.Linear(128 , 10)
    
    def forward(self , x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1 , 32 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = DigitCNN()
print(f"\n CNN Architecture {model}")


#====train ======

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr = 0.001)

print("\n training CNN")
for epoch in range(3):
    running_loss = 0.0
    for i ,(images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs , labels)
        loss.backward()
        optimizer.step()
        running_loss+= loss.item()
    print(f"epoch {epoch+1}/3 - loss :{running_loss/len(train_loader):.4f}")

# ==== test=====

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs , 1)
        total += labels.size(0)
        correct+= (predicted == labels).sum().item()

print(f"\n Test accuracy :{100*correct/total:>2f}%")
    

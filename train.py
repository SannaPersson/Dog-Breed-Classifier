import torch
import numpy as np
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
from torch.utils.data import Dataset, DataLoader  # Gives easier dataset managment and creates mini batches
from data import get_loaders
from utils import save_checkpoint, load_checkpoint, trace_model
from model import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(seed=0)

# Hyperparameters
learning_rate = 1e-3
batch_size = 10
num_epochs = 20

# Load Data
train_transform = transforms.Compose([
    transforms.Resize((356, 356)),
    torchvision.transforms.RandomCrop(224, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
    torchvision.transforms.RandomRotation(30, resample=False, expand=False, center=None, fill=None),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomErasing(p=0.3)
]
)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]
)
train_loader, validation_loader, test_loader = get_loaders(train_transform, test_transform, batch_size)

# Model
model = Net()
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        trace_model(model)
        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}')

    checkpoint = model.state_dict()
    # Try save checkpoint
    save_checkpoint(checkpoint)
    if epoch % 5 == 0:
        trace_model(model)

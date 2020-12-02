import torch
import torchvision
import os
import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, Subset, DataLoader  # Gives easier dataset management and creates mini batches
import torchvision.transforms as transforms  # Transformations we can perform on our dataset

class DogBreedsDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        if self.transform:
            image = self.transform(image)

        label = self.dataset[index][1]
        return image, label


def get_loaders(train_transform, test_transform, batch_size=32, train_split=0.8):
    dataset = torchvision.datasets.ImageFolder(root='images/Images')

    train_dataset = DogBreedsDataset(dataset, train_transform)
    validation_dataset = DogBreedsDataset(dataset, test_transform)
    test_dataset = DogBreedsDataset(dataset, test_transform)

    # Create the index splits for training, validation and test
    split_1, split_2 = train_split, (1+train_split)/2
    indices = list(range(len(dataset)))
    split1_idx, split2_idx = int(len(dataset) * split_1), int(len(dataset) * split_2)
    np.random.shuffle(indices)
    train_idx, valid_idx, test_idx = indices[:split1_idx], indices[split1_idx:split2_idx], indices[split2_idx:]
    train_set = Subset(train_dataset, indices=train_idx)
    validation_set = Subset(validation_dataset, indices=valid_idx)
    test_set = Subset(test_dataset, indices=test_idx)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader, test_loader


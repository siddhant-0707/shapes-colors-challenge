"""
Data loading and preprocessing utilities
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import ast


# Define constants
SHAPES = ['circle', 'square', 'triangle']
COLORS = ['red', 'blue', 'green']
SHAPE_COLOR_PAIRS = [(shape, color) for shape in SHAPES for color in COLORS]


def parse_label(label_str):
    """Parse label string to list of tuples"""
    try:
        return ast.literal_eval(label_str)
    except:
        return []


def labels_to_multihot(labels):
    """Convert list of (shape, color) tuples to multi-hot encoding"""
    vector = np.zeros(len(SHAPE_COLOR_PAIRS), dtype=np.float32)
    for label in labels:
        if label in SHAPE_COLOR_PAIRS:
            idx = SHAPE_COLOR_PAIRS.index(label)
            vector[idx] = 1.0
    return vector


def multihot_to_labels(vector, threshold=0.5):
    """Convert multi-hot encoding back to list of (shape, color) tuples"""
    labels = []
    for i, val in enumerate(vector):
        if val >= threshold:
            labels.append(SHAPE_COLOR_PAIRS[i])
    return labels


class ShapesColorsDataset(Dataset):
    """PyTorch Dataset for shapes and colors images"""
    
    def __init__(self, dataframe, data_dir, transform=None):
        self.dataframe = dataframe
        self.data_dir = Path(data_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.data_dir / self.dataframe.iloc[idx]['image_path']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        labels = self.dataframe.iloc[idx]['parsed_labels']
        target = labels_to_multihot(labels)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(target, dtype=torch.float32)


class TestDataset(Dataset):
    """PyTorch Dataset for test images (no labels)"""
    
    def __init__(self, dataframe, data_dir, transform=None):
        self.dataframe = dataframe
        self.data_dir = Path(data_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.data_dir / self.dataframe.iloc[idx]['image_path']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image


def get_transforms():
    """Get train and validation transforms"""
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_dataloaders(train_data, val_data, data_dir, batch_size=32, num_workers=0):
    """Create train and validation dataloaders"""
    train_transform, val_transform = get_transforms()
    
    train_dataset = ShapesColorsDataset(train_data, data_dir, transform=train_transform)
    val_dataset = ShapesColorsDataset(val_data, data_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


#!/usr/bin/env python3
"""
Simple training script for chest X-ray classification that works without the full ReXGradient dataset.
This script creates a minimal working example for testing the training pipeline.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from utils.model_utils import MultiLabelResNet, TARGET_PATHOLOGIES

class SimpleChestXrayDataset(Dataset):
    """Simple dataset for testing - creates synthetic data if no real data is available."""
    
    def __init__(self, num_samples=1000, image_size=(224, 224), num_classes=14):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Create synthetic data
        self.data = []
        for i in range(num_samples):
            # Create random image data (in real scenario, this would be actual X-ray images)
            image = torch.randn(3, *image_size)  # Random tensor as placeholder
            # Create random labels (binary multi-label)
            labels = torch.randint(0, 2, (num_classes,)).float()
            
            self.data.append({
                'image': image,
                'labels': labels
            })
        
        # Transform for training
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample['image']
        labels = sample['labels']
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, labels

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy (for multi-label, we use threshold)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predicted == labels).all(dim=1).sum().item()
        total += labels.size(0)
        
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Calculate accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).all(dim=1).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description="Simple Chest X-ray Classification Training")
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet101",
                       choices=["resnet101", "resnext101_32x8d", "efficientnet_b4"],
                       help="CNN backbone architecture")
    parser.add_argument("--num_classes", type=int, default=14,
                       help="Number of pathology classes")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    
    # Dataset parameters
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of synthetic samples to generate")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Image size for training")
    
    # Other parameters
    parser.add_argument("--model_save_dir", type=str, default="models",
                       help="Directory to save models")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Log interval")
    
    args = parser.parse_args()
    
    print("Starting simple training...")
    print(f"Arguments: {args}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.model_save_dir, exist_ok=True)
    
    # Create datasets
    print("Creating synthetic datasets...")
    train_dataset = SimpleChestXrayDataset(
        num_samples=args.num_samples,
        image_size=(args.image_size, args.image_size),
        num_classes=args.num_classes
    )
    
    val_dataset = SimpleChestXrayDataset(
        num_samples=200,
        image_size=(args.image_size, args.image_size),
        num_classes=args.num_classes
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print(f"Creating model with backbone: {args.backbone}")
    model = MultiLabelResNet(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=True
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'args': args
            }, os.path.join(args.model_save_dir, 'best_model.pth'))
            print(f"New best model saved with val_loss: {val_loss:.4f}")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'args': args,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }, os.path.join(args.model_save_dir, 'final_model.pth'))
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation accuracy: {val_acc:.2f}%")
    print(f"Models saved in: {args.model_save_dir}")

if __name__ == "__main__":
    main()


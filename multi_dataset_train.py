#!/usr/bin/env python3
"""
Multi-Dataset Training Script for Chest X-ray Classification
This script loads and combines ALL available datasets for comprehensive training.
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
import glob
warnings.filterwarnings('ignore')

# Import custom modules
from utils.model_utils import MultiLabelResNet, TARGET_PATHOLOGIES

class MultiDatasetChestXrayDataset(Dataset):
    """Dataset class that can handle multiple chest X-ray datasets."""
    
    def __init__(self, metadata_df, image_base_dir, is_train=True, image_size=(224, 224)):
        self.metadata_df = metadata_df
        self.image_base_dir = image_base_dir
        self.image_size = image_size
        self.is_train = is_train
        
        # Create transforms
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size[0] + 32, image_size[1] + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        
        # Get image path
        image_path = row['Image Path']
        full_path = os.path.join(self.image_base_dir, image_path)
        
        # Load image
        image = None
        try:
            if not os.path.exists(full_path):
                # Try alternative paths
                alt_paths = [
                    os.path.join(self.image_base_dir, 'images', image_path),
                    os.path.join(self.image_base_dir, 'images_normalized', image_path),
                    os.path.join(self.image_base_dir, 'sample', image_path)
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        full_path = alt_path
                        break
                else:
                    # Create a dummy image if not found
                    image = Image.new('RGB', self.image_size, color='black')
            
            if image is None:
                image = Image.open(full_path).convert('RGB')
        except Exception as e:
            # Create a dummy image if loading fails
            image = Image.new('RGB', self.image_size, color='black')
        
        # Get labels
        labels = row['Harmonized Labels']
        if isinstance(labels, str):
            # Convert string to array
            label_array = np.array([int(x) for x in labels.split()])
        else:
            label_array = np.array(labels)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label_array, dtype=torch.float32)

def load_nih_dataset(dataset_dir):
    """Load NIH ChestX-ray14 dataset."""
    print("Loading NIH ChestX-ray14 dataset...")
    
    nih_path = os.path.join(dataset_dir, "Data_Entry_2017.csv")
    if not os.path.exists(nih_path):
        print("NIH dataset not found")
        return None
    
    nih_df = pd.read_csv(nih_path)
    
    # Create harmonized labels
    def harmonize_nih_labels(finding_labels):
        if pd.isna(finding_labels) or finding_labels == 'No Finding':
            return ' '.join(['0'] * len(TARGET_PATHOLOGIES))
        
        labels = [0] * len(TARGET_PATHOLOGIES)
        finding_labels = finding_labels.lower()
        
        for i, pathology in enumerate(TARGET_PATHOLOGIES):
            if pathology.lower() in finding_labels:
                labels[i] = 1
        
        return ' '.join(map(str, labels))
    
    nih_df['Harmonized Labels'] = nih_df['Finding Labels'].apply(harmonize_nih_labels)
    nih_df['Dataset'] = 'NIH ChestX-ray14'
    nih_df['Image Path'] = nih_df['Image Index']
    
    print(f"NIH dataset loaded: {len(nih_df)} samples")
    return nih_df[['Image Path', 'Dataset', 'Harmonized Labels']]

def load_openi_dataset(dataset_dir):
    """Load OpenI dataset."""
    print("Loading OpenI dataset...")
    
    # Check for OpenI metadata files
    openi_files = [
        "chest_x_ray_images_labels_sample.csv",
        "indiana_reports.csv"
    ]
    
    openi_data = []
    
    # Load from chest_x_ray_images_labels_sample.csv
    openi_path = os.path.join(dataset_dir, "chest_x_ray_images_labels_sample.csv")
    if os.path.exists(openi_path):
        openi_df = pd.read_csv(openi_path)
        
        def harmonize_openi_labels(labels_str):
            if pd.isna(labels_str) or labels_str == "['normal']":
                return ' '.join(['0'] * len(TARGET_PATHOLOGIES))
            
            labels = [0] * len(TARGET_PATHOLOGIES)
            labels_str = str(labels_str).lower()
            
            # Map OpenI labels to target pathologies
            label_mapping = {
                'cardiomegaly': 'Cardiomegaly',
                'consolidation': 'Consolidation',
                'edema': 'Edema',
                'effusion': 'Effusion',
                'emphysema': 'Emphysema',
                'fibrosis': 'Fibrosis',
                'hernia': 'Hernia',
                'infiltration': 'Infiltration',
                'mass': 'Mass',
                'nodule': 'Nodule',
                'pleural_thickening': 'Pleural_Thickening',
                'pneumonia': 'Pneumonia',
                'pneumothorax': 'Pneumothorax',
                'atelectasis': 'Atelectasis'
            }
            
            for openi_label, target_label in label_mapping.items():
                if openi_label in labels_str and target_label in TARGET_PATHOLOGIES:
                    idx = TARGET_PATHOLOGIES.index(target_label)
                    labels[idx] = 1
            
            return ' '.join(map(str, labels))
        
        openi_df['Harmonized Labels'] = openi_df['Labels'].apply(harmonize_openi_labels)
        openi_df['Dataset'] = 'OpenI'
        openi_df['Image Path'] = openi_df['ImageID']
        
        openi_data.append(openi_df[['Image Path', 'Dataset', 'Harmonized Labels']])
    
    if openi_data:
        combined_openi = pd.concat(openi_data, ignore_index=True)
        print(f"OpenI dataset loaded: {len(combined_openi)} samples")
        return combined_openi
    else:
        print("OpenI dataset not found")
        return None

def load_mimic_dataset(dataset_dir):
    """Load MIMIC-CXR dataset."""
    print("Loading MIMIC-CXR dataset...")
    
    mimic_path = os.path.join(dataset_dir, "cxr_df.csv")
    if not os.path.exists(mimic_path):
        print("MIMIC-CXR dataset not found")
        return None
    
    # Load a sample of MIMIC data (it's very large)
    mimic_df = pd.read_csv(mimic_path, nrows=10000)  # Load first 10k samples
    
    def extract_labels_from_text(text):
        if pd.isna(text):
            return ' '.join(['0'] * len(TARGET_PATHOLOGIES))
        
        text = str(text).lower()
        labels = [0] * len(TARGET_PATHOLOGIES)
        
        # Simple keyword matching for pathology detection
        pathology_keywords = {
            'Cardiomegaly': ['cardiomegaly', 'enlarged heart', 'heart size'],
            'Consolidation': ['consolidation', 'consolidated'],
            'Edema': ['edema', 'pulmonary edema'],
            'Effusion': ['effusion', 'pleural effusion'],
            'Emphysema': ['emphysema', 'copd'],
            'Fibrosis': ['fibrosis', 'fibrotic'],
            'Hernia': ['hernia'],
            'Infiltration': ['infiltration', 'infiltrate'],
            'Mass': ['mass', 'lesion'],
            'Nodule': ['nodule', 'nodular'],
            'Pleural_Thickening': ['pleural thickening', 'thickening'],
            'Pneumonia': ['pneumonia', 'pneumonic'],
            'Pneumothorax': ['pneumothorax'],
            'Atelectasis': ['atelectasis', 'collapse']
        }
        
        for i, pathology in enumerate(TARGET_PATHOLOGIES):
            if pathology in pathology_keywords:
                for keyword in pathology_keywords[pathology]:
                    if keyword in text:
                        labels[i] = 1
                        break
        
        return ' '.join(map(str, labels))
    
    mimic_df['Harmonized Labels'] = mimic_df['text'].apply(extract_labels_from_text)
    mimic_df['Dataset'] = 'MIMIC-CXR'
    mimic_df['Image Path'] = mimic_df['path'].apply(lambda x: os.path.basename(x) if pd.notna(x) else '')
    
    print(f"MIMIC-CXR dataset loaded: {len(mimic_df)} samples")
    return mimic_df[['Image Path', 'Dataset', 'Harmonized Labels']]

def load_all_datasets(dataset_dir):
    """Load all available datasets."""
    print("Loading all available datasets...")
    
    all_datasets = []
    
    # Load NIH dataset
    nih_df = load_nih_dataset(dataset_dir)
    if nih_df is not None:
        all_datasets.append(nih_df)
    
    # Load OpenI dataset
    openi_df = load_openi_dataset(dataset_dir)
    if openi_df is not None:
        all_datasets.append(openi_df)
    
    # Load MIMIC-CXR dataset
    mimic_df = load_mimic_dataset(dataset_dir)
    if mimic_df is not None:
        all_datasets.append(mimic_df)
    
    if not all_datasets:
        print("No datasets found!")
        return None
    
    # Combine all datasets
    combined_df = pd.concat(all_datasets, ignore_index=True)
    print(f"Total samples from all datasets: {len(combined_df)}")
    
    # Remove samples with invalid labels
    valid_df = combined_df[combined_df['Harmonized Labels'].str.len() > 0]
    print(f"Valid samples after filtering: {len(valid_df)}")
    
    return valid_df

def create_data_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create train/validation/test splits."""
    print("Creating data splits...")
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    n_samples = len(df)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # Create splits
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return train_df, val_df, test_df

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None):
    """Train for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Use mixed precision if available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy with threshold optimization
        with torch.no_grad():
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
    parser = argparse.ArgumentParser(description="Multi-Dataset Chest X-ray Classification Training")
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="convnext_large",
                       choices=["resnet101", "resnext101_32x8d", "efficientnet_b4", "convnext_large"],
                       help="CNN backbone architecture")
    parser.add_argument("--num_classes", type=int, default=14,
                       help="Number of pathology classes")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size (reduced for ConvNeXT Large)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (lower for ConvNeXT)")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of data loading workers")
    
    # Dataset parameters
    parser.add_argument("--dataset_dir", type=str, default="../Dataset",
                       help="Directory containing all datasets")
    parser.add_argument("--image_size", type=int, default=384,
                       help="Image size for training (larger for ConvNeXT)")
    
    # Other parameters
    parser.add_argument("--model_save_dir", type=str, default="models",
                       help="Directory to save models")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Log interval")
    
    args = parser.parse_args()
    
    print("Starting multi-dataset training...")
    print(f"Arguments: {args}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.model_save_dir, exist_ok=True)
    
    # Load all datasets
    combined_df = load_all_datasets(args.dataset_dir)
    if combined_df is None:
        print("Failed to load any datasets!")
        return
    
    # Create data splits
    train_df, val_df, test_df = create_data_splits(combined_df)
    
    # Save splits
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train_metadata.csv", index=False)
    val_df.to_csv("data/processed/val_metadata.csv", index=False)
    test_df.to_csv("data/processed/test_metadata.csv", index=False)
    
    # Create datasets
    train_dataset = MultiDatasetChestXrayDataset(
        train_df, 
        args.dataset_dir, 
        is_train=True,
        image_size=(args.image_size, args.image_size)
    )
    
    val_dataset = MultiDatasetChestXrayDataset(
        val_df, 
        args.dataset_dir, 
        is_train=False,
        image_size=(args.image_size, args.image_size)
    )
    
    test_dataset = MultiDatasetChestXrayDataset(
        test_df, 
        args.dataset_dir, 
        is_train=False,
        image_size=(args.image_size, args.image_size)
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print(f"Creating model with backbone: {args.backbone}")
    model = MultiLabelResNet(
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=True
    ).to(device)
    
    # Loss and optimizer with advanced techniques for high accuracy
    criterion = nn.BCEWithLogitsLoss()
    
    # Use different learning rates for different parts of the model
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name or 'fc' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.learning_rate * 0.1},  # Lower LR for pretrained backbone
        {'params': classifier_params, 'lr': args.learning_rate}       # Higher LR for new classifier
    ], weight_decay=1e-4)
    
    # Advanced scheduler with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=args.learning_rate * 0.01
    )
    
    # Mixed precision training for better performance
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Progressive unfreezing for better accuracy
    def unfreeze_layers(model, epoch):
        """Progressively unfreeze more layers as training progresses."""
        if epoch == 20:  # After 20 epochs, unfreeze some backbone layers
            for name, param in model.named_parameters():
                if 'base_model.features.6' in name or 'base_model.features.7' in name:
                    param.requires_grad = True
            print("Unfroze backbone layers 6-7")
        elif epoch == 40:  # After 40 epochs, unfreeze more layers
            for name, param in model.named_parameters():
                if 'base_model.features.4' in name or 'base_model.features.5' in name:
                    param.requires_grad = True
            print("Unfroze backbone layers 4-5")
        elif epoch == 60:  # After 60 epochs, unfreeze all layers
            for param in model.parameters():
                param.requires_grad = True
            print("Unfroze all backbone layers")
    
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Progressive unfreezing
        unfreeze_layers(model, epoch)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1, scaler)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
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
    
    # Test the model
    print("\nTesting model...")
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'args': args,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }, os.path.join(args.model_save_dir, 'final_model.pth'))
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation accuracy: {val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Models saved in: {args.model_save_dir}")

if __name__ == "__main__":
    main()

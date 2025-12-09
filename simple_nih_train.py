"""
Simplified training script for NIH ChestX-ray14 dataset.
This script handles the full dataset with proper data loading and training.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')

# Import specific components to avoid tensorboard dependency
import sys
sys.path.insert(0, os.path.dirname(__file__))
import torchvision.models as models

# Define TARGET_PATHOLOGIES directly to avoid importing from model_utils
TARGET_PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# Define MultiLabelResNet directly to avoid importing model_utils
class MultiLabelResNet(nn.Module):
    def __init__(self, num_classes, backbone='resnet101', pretrained=True):
        super(MultiLabelResNet, self).__init__()
        if backbone == 'resnet101':
            self.base_model = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnext101_32x8d':
            self.base_model = models.resnext101_32x8d(pretrained=pretrained)
        elif backbone == 'efficientnet_b4':
            self.base_model = models.efficientnet_b4(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace the final classification layer for multi-label output
        if hasattr(self.base_model, 'fc'):  # ResNet/ResNeXt
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, num_classes)
            )
        elif hasattr(self.base_model, 'classifier'):  # EfficientNet
            in_features = self.base_model.classifier[-1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError(f"Unknown model architecture for backbone: {backbone}")
    
    def forward(self, x):
        return self.base_model(x)

from utils.preprocessing import create_transforms, TARGET_IMAGE_SIZE


class NIHChestXrayDataset(Dataset):
    """Dataset class specifically for NIH ChestX-ray14"""
    
    def __init__(self, dataframe, images_base_dir, is_train=True):
        """
        Args:
            dataframe: DataFrame with 'Image Index' and 'Finding Labels' columns
            images_base_dir: Base directory containing images_001 through images_012 folders
            is_train: Whether to apply training augmentations
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.images_base_dir = images_base_dir
        self.transform = create_transforms(is_train=is_train, image_size=TARGET_IMAGE_SIZE)
        self.labels = self._process_labels()
        
    def _process_labels(self):
        """Convert Finding Labels to multi-hot encoded labels"""
        labels_list = []
        for _, row in self.dataframe.iterrows():
            finding_labels = row['Finding Labels']
            label_vector = np.zeros(len(TARGET_PATHOLOGIES), dtype=np.float32)
            
            if pd.notna(finding_labels) and finding_labels != 'No Finding':
                findings = finding_labels.split('|')
                for finding in findings:
                    if finding in TARGET_PATHOLOGIES:
                        idx = TARGET_PATHOLOGIES.index(finding)
                        label_vector[idx] = 1.0
            
            labels_list.append(label_vector)
        
        return np.array(labels_list)
    
    def _find_image_path(self, image_name):
        """Search for image across images_001 to images_012 directories"""
        # Try images_001 through images_012
        for i in range(1, 13):
            folder_name = f"images_{i:03d}"
            # Check in both direct folder and images subfolder
            path1 = os.path.join(self.images_base_dir, folder_name, image_name)
            path2 = os.path.join(self.images_base_dir, folder_name, "images", image_name)
            
            if os.path.exists(path1):
                return path1
            if os.path.exists(path2):
                return path2
        
        return None
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_name = self.dataframe.iloc[idx]['Image Index']
        image_path = self._find_image_path(image_name)
        
        if image_path is None:
            # Return a blank image if not found
            print(f"Warning: Image {image_name} not found, using blank image")
            image = np.zeros((TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3), dtype=np.uint8)
        else:
            # Load image
            try:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                image = np.zeros((TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3), dtype=np.uint8)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / num_batches


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} - Validation')
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Collect predictions for metrics
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Calculate accuracy
    accuracy = (all_preds == all_labels).float().mean().item()
    
    # Calculate per-class metrics
    per_class_acc = []
    for i in range(len(TARGET_PATHOLOGIES)):
        class_correct = (all_preds[:, i] == all_labels[:, i]).float().mean().item()
        per_class_acc.append(class_correct)
    
    avg_val_loss = running_loss / len(val_loader)
    
    return avg_val_loss, accuracy, per_class_acc


def main():
    parser = argparse.ArgumentParser(description="Train on NIH ChestX-ray14 Dataset")
    
    # Data parameters
    parser.add_argument("--dataset_csv", type=str, default="../Dataset/Data_Entry_2017.csv",
                       help="Path to NIH dataset CSV")
    parser.add_argument("--images_dir", type=str, default="../Dataset",
                       help="Base directory containing images_001 through images_012")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to use (for testing), None for all")
    parser.add_argument("--exclude_image", type=str, default="00000001_000.png",
                       help="Image to exclude for testing")
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet101",
                       choices=["resnet101", "resnext101_32x8d", "efficientnet_b4"],
                       help="CNN backbone architecture")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=15,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size (reduced for stability)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    
    # Directories
    parser.add_argument("--model_save_dir", type=str, default="models",
                       help="Directory to save models")
    
    args = parser.parse_args()
    
    print("="*80)
    print("NIH ChestX-ray14 Training Script")
    print("="*80)
    print(f"Configuration:")
    print(f"  Dataset CSV: {args.dataset_csv}")
    print(f"  Images Dir: {args.images_dir}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print("="*80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset_csv}...")
    df = pd.read_csv(args.dataset_csv)
    print(f"Total samples in CSV: {len(df)}")
    
    # Exclude test image
    if args.exclude_image:
        df = df[df['Image Index'] != args.exclude_image]
        print(f"Excluded {args.exclude_image} for testing")
        print(f"Remaining samples: {len(df)}")
    
    # Limit samples if specified
    if args.max_samples:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42)
        print(f"Using {len(df)} samples for training")
    
    # Split dataset
    # Check for official split files
    train_val_list_path = os.path.join(args.images_dir, 'train_val_list.txt')
    test_list_path = os.path.join(args.images_dir, 'test_list.txt')
    
    if os.path.exists(train_val_list_path) and os.path.exists(test_list_path):
        print(f"\nUsing official split files found in {args.images_dir}")
        
        # Read split files
        with open(train_val_list_path, 'r') as f:
            train_val_list = set(line.strip() for line in f.readlines())
        
        with open(test_list_path, 'r') as f:
            test_list = set(line.strip() for line in f.readlines())
            
        # Filter dataframe based on splits
        train_val_df = df[df['Image Index'].isin(train_val_list)]
        test_df = df[df['Image Index'].isin(test_list)]
        
        # Split train_val into train and val (80/20)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
        
        print(f"  Train: {len(train_df)} samples (from official train_val_list)")
        print(f"  Val: {len(val_df)} samples (from official train_val_list)")
        print(f"  Test: {len(test_df)} samples (from official test_list)")
        
    else:
        print("\nOfficial split files not found. Creating random train/val/test split (70/15/15)...")
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = NIHChestXrayDataset(train_df, args.images_dir, is_train=True)
    val_dataset = NIHChestXrayDataset(val_df, args.images_dir, is_train=False)
    test_dataset = NIHChestXrayDataset(test_df, args.images_dir, is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print(f"\nCreating model with {args.backbone} backbone...")
    model = MultiLabelResNet(
        num_classes=len(TARGET_PATHOLOGIES),
        backbone=args.backbone,
        pretrained=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Create model save directory
    os.makedirs(args.model_save_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        
        # Validate
        val_loss, val_accuracy, per_class_acc = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Print top 5 per-class accuracies
        class_accs = [(TARGET_PATHOLOGIES[i], per_class_acc[i]) for i in range(len(TARGET_PATHOLOGIES))]
        class_accs.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top 5 Class Accuracies:")
        for name, acc in class_accs[:5]:
            print(f"    {name}: {acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_accuracy
            
            model_path = os.path.join(args.model_save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'backbone': args.backbone,
                'num_classes': len(TARGET_PATHOLOGIES)
            }, model_path)
            
            print(f"\n  ✓ New best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    # Final evaluation
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {os.path.join(args.model_save_dir, 'best_model.pth')}")
    
    # Test on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy, test_per_class_acc = validate_epoch(model, test_loader, criterion, device, args.epochs)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\n" + "="*80)
    print("All Done! Model is ready for use in the Website.")
    print("="*80)


if __name__ == "__main__":
    main()

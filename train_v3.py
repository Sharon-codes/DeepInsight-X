# Train V3 - Optimized for class imbalance
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
import warnings

warnings.filterwarnings('ignore')

# Configuration
TARGET_PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]
TARGET_IMAGE_SIZE = (224, 224)

# Enhanced Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3, reduction='mean'):  # gamma=3 for stronger focus
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss

def create_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5), # Increased limits
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3), # Cutout regularization
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

class RobustChestXrayDataset(Dataset):
    def __init__(self, metadata_df, image_dirs, is_train=True):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.image_dirs = image_dirs
        self.transform = create_transforms(is_train)
        
        self.labels = self._parse_all_labels()
        
    def _parse_all_labels(self):
        parsed_labels = []
        for idx, row in self.metadata_df.iterrows():
            label_vec = np.zeros(len(TARGET_PATHOLOGIES), dtype=np.float32)
            raw_label = row.get('Harmonized Labels', '')
            
            if isinstance(raw_label, str) and all(c in '01 ' for c in raw_label):
                parts = raw_label.split()
                if len(parts) == len(TARGET_PATHOLOGIES):
                    label_vec = np.array([float(x) for x in parts], dtype=np.float32)
            
            parsed_labels.append(label_vec)
                
        return np.array(parsed_labels)

    def __getitem__(self, idx):
        # Prioritize absolute path if available
        abs_path = str(self.metadata_df.iloc[idx].get('Processed Image Path', ''))
        
        image = None
        if os.path.exists(abs_path):
             try:
                image = cv2.imread(abs_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
             except:
                pass
        
        # Fallback to searching in image_dirs if absolute path didn't work
        if image is None:
            img_name = os.path.basename(str(self.metadata_df.iloc[idx].get('Image Index', '')))
            for img_dir in self.image_dirs:
                path = os.path.join(img_dir, img_name)
                if os.path.exists(path):
                    try:
                        image = cv2.imread(path)
                        if image is not None:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            break
                    except:
                        pass
        
        if image is None:
            image = np.zeros((TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3), dtype=np.uint8)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label, idx  # Return index for sample weighting

    def __len__(self):
        return len(self.metadata_df)

class MultiLabelModel(nn.Module):
    def __init__(self, num_classes, backbone='convnext_large'):
        super(MultiLabelModel, self).__init__()
        if backbone == 'convnext_large':
            self.base_model = models.convnext_large(pretrained=True)
            in_features = self.base_model.classifier[2].in_features
            self.base_model.classifier[2] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

def calculate_metrics_with_optimal_thresholds(y_true, y_prob):
    """Calculate metrics with per-class optimal thresholds"""
    num_classes = y_true.shape[1]
    
    # Find optimal threshold per class
    optimal_thresholds = []
    for i in range(num_classes):
        if y_true[:, i].sum() == 0:
            optimal_thresholds.append(0.5)
            continue
            
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (y_prob[:, i] > thresh).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        optimal_thresholds.append(best_thresh)
    
    # Apply optimal thresholds
    y_pred = np.zeros_like(y_prob)
    for i in range(num_classes):
        y_pred[:, i] = (y_prob[:, i] > optimal_thresholds[i]).astype(int)
    
    
    # Calculate Hamming Accuracy (Label-based accuracy)
    # This is more intuitive for multi-label: (TP + TN) / Total
    correct_labels = (y_true == y_pred).sum()
    total_labels = y_true.size
    hamming_acc = correct_labels / total_labels

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred), # Strict subset accuracy (Exact Match)
        'hamming_accuracy': hamming_acc,            # Per-label accuracy (much higher)
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'auroc_macro': roc_auc_score(y_true, y_prob, average='macro'),
        'optimal_thresholds': optimal_thresholds
    }
    return metrics

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Scheduler stepped at epoch end
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            all_probs.append(torch.sigmoid(outputs).cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    
    metrics = calculate_metrics_with_optimal_thresholds(all_targets, all_probs)
    metrics['loss'] = running_loss / len(loader)
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5) # Lower Learning rate for stability
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading dataset...")
    full_df = pd.read_csv("data/processed/train_metadata.csv")
    
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)
    
    image_dirs = ["data/processed/images"]
    train_ds = RobustChestXrayDataset(train_df, image_dirs, is_train=True)
    val_ds = RobustChestXrayDataset(val_df, image_dirs, is_train=False)
    
    # Calculate class weights for sampling
    class_counts = train_ds.labels.sum(axis=0)
    class_weights = 1.0 / (class_counts + 1)  # +1 to avoid division by zero
    
    # Sample weights (emphasize rare diseases) - DISABLED because it causes over-prediction and low AUROC initially
    # sample_weights = []
    # for i in range(len(train_ds)):
    #     # Weight samples by rarest disease present
    #     labels = train_ds.labels[i]
    #     active_classes = np.where(labels == 1)[0]
    #     if len(active_classes) > 0:
    #         weight = class_weights[active_classes].max()  # Use rarest disease
    #     else:
    #         weight = 1.0
    #     sample_weights.append(weight)
    
    # sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Use standard shuffling instead of weighted sampler
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    print("Creating model...")
    model = MultiLabelModel(len(TARGET_PATHOLOGIES)).to(device)
    
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05) # Increased WD for regularization
    scaler = GradScaler()
    
    # Reliable plateau breaking
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.2, 
        patience=2, 
        verbose=True
    )
    
    os.makedirs("models", exist_ok=True)
    best_auroc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val AUROC: {val_metrics['auroc_macro']:.4f}")
        print(f"Val Exact Match Acc: {val_metrics['accuracy']:.4f} (Strict)")
        print(f"Val Label Accuracy:  {val_metrics['hamming_accuracy']:.4f} (Avg per disease)")
        print(f"Val Precision: {val_metrics['precision_micro']:.4f}")
        print(f"Val F1: {val_metrics['f1_micro']:.4f}")
        print(f"Val Recall: {val_metrics['recall_micro']:.4f}")
        
        # Step scheduler if plateau
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics['auroc_macro'])
        else:
            scheduler.step()
        
        if val_metrics['auroc_macro'] > best_auroc:
            best_auroc = val_metrics['auroc_macro']
            torch.save(model.state_dict(), "models/best_model_v3.pth")
            print("✓ Saved new best model")

if __name__ == "__main__":
    main()

# train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from utils.model_utils import MultiLabelResNet, TARGET_PATHOLOGIES
from utils.data_loader import ChestXrayDataset
from utils.preprocessing import create_transforms, TARGET_IMAGE_SIZE
from utils.rexgradient_loader import process_rexgradient_dataset
from utils.report_generator import ClassificationReportGenerator

def create_synthetic_dataset(num_samples=5000):
    """
    Create a synthetic dataset for training when real datasets are not available.
    
    Args:
        num_samples (int): Number of synthetic samples to create
        
    Returns:
        pd.DataFrame: Synthetic dataset metadata
    """
    print(f"Creating synthetic dataset with {num_samples} samples...")
    
    # Create synthetic data
    synthetic_data = []
    os.makedirs("data/processed/images", exist_ok=True)
    
    for i in range(num_samples):
        # Create random image (in real scenario, this would be actual X-ray images)
        # For now, we'll create a placeholder entry
        filename = f"synthetic_{i:06d}.png"
        image_path = f"data/processed/images/{filename}"
        
        # Create random multi-label annotations
        # Simulate realistic pathology distribution
        labels = np.random.choice([0, 1], size=len(TARGET_PATHOLOGIES), p=[0.8, 0.2])
        
        # Create metadata entry
        metadata_entry = {
            'Image Index': filename,
            'Dataset': 'Synthetic',
            'Split': 'train' if i < num_samples * 0.7 else ('val' if i < num_samples * 0.85 else 'test'),
            'Processed Image Path': image_path,
            'Original Labels': TARGET_PATHOLOGIES,
            'Report': f'Synthetic chest X-ray report {i}',
            'Harmonized Labels': ' '.join(map(str, labels))
        }
        
        synthetic_data.append(metadata_entry)
    
    # Create dataframe
    synthetic_df = pd.DataFrame(synthetic_data)
    
    # Save synthetic metadata
    os.makedirs("data/processed", exist_ok=True)
    synthetic_df.to_csv("data/processed/synthetic_metadata.csv", index=False)
    print(f"Synthetic dataset created with {len(synthetic_df)} samples")
    
    return synthetic_df

class SyntheticChestXrayDataset(torch.utils.data.Dataset):
    """Dataset class for synthetic X-ray images."""
    
    def __init__(self, metadata_csv, image_dir, is_train=True):
        self.metadata = pd.read_csv(metadata_csv)
        self.image_dir = image_dir
        self.is_train = is_train
        self.transform = create_transforms(is_train=is_train)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get metadata
        row = self.metadata.iloc[idx]
        
        # Get labels
        labels_str = row['Harmonized Labels']
        labels = torch.tensor([int(x) for x in labels_str.split()], dtype=torch.float32)
        
        # Generate synthetic image
        img_array = np.random.randint(0, 255, (TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3), dtype=np.uint8)
        image = Image.fromarray(img_array)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, labels

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in multi-label classification.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Apply sigmoid to inputs to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate focal loss
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint."""
        self.best_weights = model.state_dict().copy()

def load_and_prepare_data(args):
    """
    Load and prepare datasets for training using HuggingFace datasets.
    Supports NIH ChestX-ray14 (images) and ReXGradient-160K (text reports).
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    from datasets import load_dataset
    from PIL import Image
    import re
    
    print("Loading and preparing datasets...")
    
    # Load NIH ChestX-ray14 from HuggingFace
    nih_df = None
    nih_metadata_cache = "data/processed/nih_full_metadata.csv"
    
    # Check if we already have processed NIH data
    if os.path.exists(nih_metadata_cache):
        print(f"✓ Found cached NIH metadata at {nih_metadata_cache}")
        print("  Loading from cache (skipping image processing)...")
        nih_df = pd.read_csv(nih_metadata_cache)
        
        # Validate cache: Check if we have any positive labels
        # The 'Harmonized Labels' column contains strings like "0 0 1 0..."
        # If all rows are "0 0 0...", then the cache is invalid.
        has_findings = False
        for labels in nih_df['Harmonized Labels'].head(1000):
            if '1' in str(labels):
                has_findings = True
                break
        
        if not has_findings and len(nih_df) > 0:
            print("⚠️ DETECTED INVALID CACHE: All checked labels are zero (no findings).")
            print("  Deleting corrupted cache and forcing re-processing...")
            os.remove(nih_metadata_cache)
            nih_df = None # Force re-processing
        else:
            print(f"✓ Loaded {len(nih_df)} NIH samples from cache")
    else:
        # Process NIH dataset from HuggingFace
        try:
            print("Loading NIH ChestX-ray14 dataset from HuggingFace...")
            # NIH dataset has configs: 'image-classification' for multi-label classification
            nih_dataset = load_dataset('alkzar90/NIH-Chest-X-ray-dataset', 
                                       name='image-classification',  # Use image-classification config
                                       split='train', 
                                       trust_remote_code=True)
            print(f"NIH dataset loaded: {len(nih_dataset)} samples")
            
            # Convert to DataFrame
            nih_data = []
            print(f"Processing {len(nih_dataset)} NIH samples (this may take a while)...")
            os.makedirs("data/processed/images", exist_ok=True)
            
            for idx, sample in enumerate(tqdm(nih_dataset, desc="Processing NIH images")):
                # Save image
                img_path = f"data/processed/images/nih_{idx:06d}.png"
                sample['image'].save(img_path)
                
                # Inspect first sample to find label key
                if idx == 0:
                    print(f"Sample keys: {sample.keys()}")
                
                # Get labels - handle different formats
                finding_labels = ""
                labels_vec = ['0'] * len(TARGET_PATHOLOGIES)
                
                if 'labels' in sample:
                    # HuggingFace image-classification format (list of integers)
                    label_indices = sample['labels']
                    # Get class names from dataset features if available
                    if hasattr(nih_dataset.features['labels'], 'names'):
                        class_names = nih_dataset.features['labels'].names
                        finding_labels_list = [class_names[i] for i in label_indices]
                        finding_labels = '|'.join(finding_labels_list)
                        
                        # Map to our TARGET_PATHOLOGIES
                        for i, pathology in enumerate(TARGET_PATHOLOGIES):
                            # Handle potential spelling differences (e.g. "Pleural Thickening" vs "Pleural_Thickening")
                            pathology_clean = pathology.replace('_', ' ')
                            for lbl in finding_labels_list:
                                if pathology_clean.lower() == lbl.lower() or pathology.lower() == lbl.lower():
                                    labels_vec[i] = '1'
                    else:
                        # Fallback if names not available (unlikely for this dataset)
                        finding_labels = str(label_indices)
                        
                elif 'Finding Labels' in sample:
                    # Original CSV format
                    finding_labels = sample['Finding Labels']
                    for i, pathology in enumerate(TARGET_PATHOLOGIES):
                        pathology_clean = pathology.replace('_', ' ')
                        if pathology in finding_labels or pathology_clean in finding_labels:
                            labels_vec[i] = '1'
                
                labels = ' '.join(labels_vec)
                
                nih_data.append({
                    'Image Index': f"nih_{idx:06d}.png",
                    'Dataset': 'NIH ChestX-ray14',
                    'Split': 'train',
                    'Processed Image Path': img_path,
                    'Finding Labels': finding_labels,
                    'Harmonized Labels': labels
                })
                
                # Log progress every 5000 samples
                if (idx + 1) % 5000 == 0:
                    print(f"  → Processed {idx + 1}/{len(nih_dataset)} samples ({(idx + 1) / len(nih_dataset) * 100:.1f}%)")
            
            nih_df = pd.DataFrame(nih_data)
            # Save the full metadata for future use
            nih_df.to_csv(nih_metadata_cache, index=False)
            print(f"✓ Processed {len(nih_df)} NIH samples and cached metadata")
        except Exception as e:
            print(f"Error loading NIH dataset: {e}. Continuing without it.")
    
    # Load ReXGradient-160K text reports
    rex_df = None
    try:
        print("Loading ReXGradient-160K text reports from HuggingFace...")
        rex_dataset = load_dataset("rajpurkarlab/ReXGradient-160K", split="train")
        print(f"ReXGradient dataset loaded: {len(rex_dataset)} samples")
        
        # Extract pathologies from text
        pathology_keywords = {
            'Atelectasis': ['atelectasis', 'collapse'],
            'Cardiomegaly': ['cardiomegaly', 'enlarged heart'],
            'Consolidation': ['consolidation'],
            'Edema': ['edema', 'pulmonary edema'],
            'Effusion': ['effusion', 'pleural effusion'],
            'Emphysema': ['emphysema'],
            'Fibrosis': ['fibrosis'],
            'Hernia': ['hernia'],
            'Infiltration': ['infiltrate', 'infiltration'],
            'Mass': ['mass', 'masses'],
            'Nodule': ['nodule'],
            'Pleural_Thickening': ['pleural thickening'],
            'Pneumonia': ['pneumonia'],
            'Pneumothorax': ['pneumothorax']
        }
        
        rex_data = []
        for sample in rex_dataset:
            text = f"{sample.get('Findings', '')} {sample.get('Impression', '')}".lower()
            labels = []
            for pathology, keywords in pathology_keywords.items():
                if any(kw in text for kw in keywords):
                    labels.append('1')
                else:
                    labels.append('0')
            
            if '1' in labels:  # Only keep samples with detected pathologies
                rex_data.append({
                    'ReportID': sample.get('id', ''),
                    'Findings': sample.get('Findings', ''),
                    'Impression': sample.get('Impression', ''),
                    'Harmonized Labels': ' '.join(labels)
                })
        
        rex_df = pd.DataFrame(rex_data)
        print(f"Processed {len(rex_df)} ReXGradient samples with pathology labels")
        # Save for future reference
        os.makedirs("data/processed", exist_ok=True)
        rex_df.to_csv("data/processed/rexgradient_text_reports.csv", index=False)
    except Exception as e:
        print(f"Error loading ReXGradient dataset: {e}. Continuing without it.")
    
    # Combine datasets
    all_datasets = []
    if nih_df is not None and len(nih_df) > 0:
        all_datasets.append(nih_df)
    
    # Check if we have any datasets
    if len(all_datasets) == 0:
        print("No real datasets available. Creating synthetic dataset for training...")
        combined_df = create_synthetic_dataset(num_samples=5000)
    else:
        combined_df = pd.concat(all_datasets, ignore_index=True)
        print(f"Total samples loaded: {len(combined_df)}")
    
    # Create stratified train/val/test split
    print("Creating stratified train/validation/test split...")
    
    train_df, temp_df = train_test_split(
        combined_df, test_size=0.3, random_state=42, stratify=None
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=None
    )
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Save splits
    print("Saving metadata splits to CSV files...")
    train_df.to_csv("data/processed/train_metadata.csv", index=False)
    val_df.to_csv("data/processed/val_metadata.csv", index=False)
    test_df.to_csv("data/processed/test_metadata.csv", index=False)
    print("✓ Metadata CSV files saved successfully")
    
    # Create datasets
    print("Creating PyTorch datasets...")
    image_dir = "data/processed/images"
    
    # Check if we're using synthetic data
    if len(all_datasets) == 0:
        train_dataset = SyntheticChestXrayDataset(
            "data/processed/train_metadata.csv", 
            image_dir, 
            is_train=True
        )
        val_dataset = SyntheticChestXrayDataset(
            "data/processed/val_metadata.csv", 
            image_dir, 
            is_train=False
        )
        test_dataset = SyntheticChestXrayDataset(
            "data/processed/test_metadata.csv", 
            image_dir, 
            is_train=False
        )
    else:
        train_dataset = ChestXrayDataset(
            "data/processed/train_metadata.csv", 
            image_dir, 
            is_train=True
        )
        val_dataset = ChestXrayDataset(
            "data/processed/val_metadata.csv", 
            image_dir, 
            is_train=False
        )
        test_dataset = ChestXrayDataset(
            "data/processed/test_metadata.csv", 
            image_dir, 
            is_train=False
        )
    
    # Create data loaders
    print("Creating data loaders...")
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
    print(f"✓ Data loaders created (batch_size={args.batch_size}, num_workers={args.num_workers})")
    
    return train_loader, val_loader, test_loader, len(TARGET_PATHOLOGIES)

def calculate_class_weights(train_loader, num_classes):
    """
    Calculate class weights for handling imbalanced data.
    
    Args:
        train_loader: Training data loader
        num_classes: Number of classes
        
    Returns:
        torch.Tensor: Class weights
    """
    print("Calculating class weights...")
    print(f"  Iterating through {len(train_loader)} batches to collect labels...")
    
    # Collect all labels
    all_labels = []
    for batch_idx, (_, labels) in enumerate(tqdm(train_loader, desc="Collecting labels")):
        all_labels.append(labels.numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"✓ Collected labels from {len(all_labels)} samples")
    
    # Calculate class weights
    class_counts = np.sum(all_labels, axis=0)
    total_samples = len(all_labels)
    
    # Calculate weights inversely proportional to class frequency
    class_weights = []
    for count in class_counts:
        if count > 0:
            weight = total_samples / (num_classes * count)
            class_weights.append(weight)
        else:
            class_weights.append(1.0)  # Default weight for classes with no samples
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class weights: {class_weights}")
    
    return class_weights

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, args):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        if batch_idx % args.log_interval == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
    
    return running_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
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
    
    # Calculate metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Calculate accuracy
    accuracy = (all_preds == all_labels).float().mean().item()
    
    return running_loss / len(val_loader), accuracy, all_preds.numpy(), all_labels.numpy()

def train_model(args):
    """Main training function."""
    print("Starting training...")
    print(f"Arguments: {args}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_loaders = load_and_prepare_data(args)
    if data_loaders is None:
        print("Failed to load data. Exiting.")
        return
    
    train_loader, val_loader, test_loader, num_classes = data_loaders
    
    # Create model
    model = MultiLabelResNet(
        num_classes=num_classes, 
        backbone=args.backbone, 
        pretrained=True
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_loader, num_classes).to(device)
    
    # Loss function
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Training loop
    best_val_loss = float('inf')
    best_accuracy = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, args)
        
        # Validate
        val_loss, val_accuracy, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Val Accuracy: {val_accuracy:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_accuracy = val_accuracy
            
            os.makedirs(args.model_save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'args': args
            }, os.path.join(args.model_save_dir, 'best_model.pth'))
            
            print(f"New best model saved! Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.6f}")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    writer.close()
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.model_save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_loss, test_accuracy, test_preds, test_labels = validate_epoch(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Accuracy: {test_accuracy:.6f}")
    
    # Generate comprehensive report
    print("\nGenerating comprehensive evaluation report...")
    
    # Get test probabilities
    model.eval()
    test_probs = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            with autocast():
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                test_probs.append(probs.cpu())
    
    test_probs = torch.cat(test_probs).numpy()
    
    # Generate report
    report_generator = ClassificationReportGenerator(output_dir=args.report_dir)
    metrics = report_generator.generate_comprehensive_report(
        test_labels, test_preds, test_probs,
        model_name=f"{args.backbone}_ChestX-ray_Model",
        dataset_name="Test Dataset"
    )
    
    # Check if target accuracy is achieved
    auroc_score = metrics['overall']['auroc']
    accuracy_score = metrics['overall']['accuracy']
    
    print("\n" + "="*50)
    print("TARGET ACCURACY CHECK")
    print("="*50)
    print(f"AUROC Score: {auroc_score:.4f} (Target: 0.86-0.92)")
    print(f"Multi-label Accuracy: {accuracy_score:.4f} (Target: 0.86-0.92)")
    
    if 0.86 <= auroc_score <= 0.92 and 0.86 <= accuracy_score <= 0.92:
        print("✅ TARGET ACCURACY ACHIEVED!")
    else:
        print("❌ Target accuracy not achieved. Consider:")
        print("   - Training for more epochs")
        print("   - Adjusting learning rate")
        print("   - Using different augmentation strategies")
        print("   - Trying different model architectures")
    
    print("Training completed!")

def main():
    parser = argparse.ArgumentParser(description="Train Chest X-ray Classification Model")
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="convnext_large",
                       choices=["resnet101", "resnext101_32x8d", "efficientnet_b4", "convnext_large"],
                       help="CNN backbone architecture")
    parser.add_argument("--num_classes", type=int, default=14,
                       help="Number of pathology classes")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    
    # Loss function
    parser.add_argument("--use_focal_loss", action="store_true",
                       help="Use focal loss instead of BCE loss")
    
    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=15,
                       help="Early stopping patience")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=50,
                       help="Log interval")
    
    # Directories
    parser.add_argument("--model_save_dir", type=str, default="models",
                       help="Directory to save models")
    parser.add_argument("--report_dir", type=str, default="reports",
                       help="Directory to save reports")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.model_save_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Start training
    train_model(args)

if __name__ == "__main__":
    main()

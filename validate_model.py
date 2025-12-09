#!/usr/bin/env python3
"""
Validation script for the trained chest X-ray classification model.
Uses the real Dataset folder and existing metadata files.
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import cv2
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
import glob # Added for glob.glob

# Import custom modules
from utils.model_utils import MultiLabelResNet, TARGET_PATHOLOGIES

class ValidationDataset(Dataset):
    """Dataset class for validation using real data from Dataset folder."""
    
    def __init__(self, metadata_file, dataset_base_dir, image_size=(224, 224)):
        self.metadata_df = pd.read_csv(metadata_file)
        self.dataset_base_dir = dataset_base_dir # Moved this line up
        # Filter out MIMIC-CXR entries if they are not available locally
        if not os.path.exists(os.path.join(self.dataset_base_dir, 'mimic_dset')):
            initial_samples = len(self.metadata_df)
            self.metadata_df = self.metadata_df[self.metadata_df['Dataset'] != 'MIMIC-CXR']
            if len(self.metadata_df) < initial_samples:
                print(f"Filtered out {initial_samples - len(self.metadata_df)} MIMIC-CXR samples from {metadata_file}")

        self.image_size = image_size
        
        self.all_image_dirs = set()
        print(f"Scanning for image directories in {dataset_base_dir}...")
        for root, dirs, files in os.walk(dataset_base_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg')):
                    self.all_image_dirs.add(root)
                    break # Only need to add the directory once
        
        self.all_image_dirs = sorted(list(self.all_image_dirs))
        print(f"Found {len(self.all_image_dirs)} image directories: {self.all_image_dirs}")
        
        # Create transforms for validation
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        
        # Get image path from metadata
        image_path = row['Image Path']
        
        # Extract the base filename without any directory prefixes or extensions
        # For NIH images, the 'Image Path' might be '00005538_000.png'. For MIMIC-CXR, it might be 's50368257.jpg'.
        # We want to extract just '00005538_000' or 's50368257'.
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Find the actual image
        image = None
        found_path = None
        
        possible_full_paths = []
        for img_dir in self.all_image_dirs:
            possible_full_paths.append(os.path.join(img_dir, f"{base_filename}.png"))
            possible_full_paths.append(os.path.join(img_dir, f"{base_filename}.jpg"))
            # Also consider cases where the filename might be nested deeper in the img_dir, if it's a very broad dir
            # For example, if img_dir is '../Dataset', and the image is in '../Dataset/some_subfolder/s50368257.jpg'
            # This would cover cases where metadata image_path doesn't contain subfolder structure
            possible_full_paths.extend(glob.glob(os.path.join(img_dir, '**', f"{base_filename}.png"), recursive=True))
            possible_full_paths.extend(glob.glob(os.path.join(img_dir, '**', f"{base_filename}.jpg"), recursive=True))

        # Prioritize shorter paths (more direct matches) over longer paths
        possible_full_paths = sorted(list(set(possible_full_paths)), key=len)

        for full_path in possible_full_paths:
            if os.path.exists(full_path):
                try:
                    image = Image.open(full_path).convert('RGB')
                    found_path = full_path
                    break
                except Exception as e:
                    pass # Continue to next path
        
        if image is None:
            raise FileNotFoundError(f"Image not found for base filename '{base_filename}' in any of the expected locations after checking all image directories.")
        
        # Get labels
        labels = row['Harmonized Labels']
        if isinstance(labels, str):
            label_array = np.array([int(x) for x in labels.split()])
        else:
            label_array = np.array(labels)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label_array, dtype=torch.float32)

def load_model(model_path, num_classes, backbone="resnext101_32x8d", device="cuda"):
    """Load a trained model from checkpoint."""
    print(f"Loading model from {model_path}")
    
    # Create model
    model = MultiLabelResNet(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation loss: {checkpoint.get('val_loss', 'unknown')}")
        print(f"Validation accuracy: {checkpoint.get('val_accuracy', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Model state dict loaded directly")
    
    model.eval()
    return model

def evaluate_model(model, data_loader, device, threshold=0.5):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    
    # Concatenate all results
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_probs).numpy()
    
    return y_true, y_pred, y_prob

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive metrics."""
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    per_class_metrics = []
    for i, pathology in enumerate(TARGET_PATHOLOGIES):
        if np.sum(y_true[:, i]) > 0:  # Only if class exists
            precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
            f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            
            try:
                auroc = roc_auc_score(y_true[:, i], y_prob[:, i])
            except:
                auroc = 0.5
            
            per_class_metrics.append({
                'Pathology': pathology,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'AUROC': auroc,
                'Support': int(np.sum(y_true[:, i]))
            })
    
    # Calculate average metrics
    avg_precision = np.mean([m['Precision'] for m in per_class_metrics])
    avg_recall = np.mean([m['Recall'] for m in per_class_metrics])
    avg_f1 = np.mean([m['F1-Score'] for m in per_class_metrics])
    avg_auroc = np.mean([m['AUROC'] for m in per_class_metrics])
    
    return {
        'accuracy': accuracy,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'avg_auroc': avg_auroc,
        'per_class_metrics': per_class_metrics
    }

def main():
    parser = argparse.ArgumentParser(description="Validate Chest X-ray Classification Model")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default="models/best_model.pth",
                       help="Path to trained model checkpoint")
    parser.add_argument("--backbone", type=str, default="resnext101_32x8d",
                       choices=["resnet101", "resnext101_32x8d", "efficientnet_b4"],
                       help="CNN backbone architecture")
    parser.add_argument("--num_classes", type=int, default=14,
                       help="Number of pathology classes")
    
    # Data parameters
    parser.add_argument("--dataset_dir", type=str, default="../Dataset",
                       help="Directory containing the Dataset folder")
    parser.add_argument("--metadata_dir", type=str, default="data/processed",
                       help="Directory containing metadata files")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of data loading workers")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for binary predictions")
    
    # Other parameters
    parser.add_argument("--report_dir", type=str, default="validation_reports",
                       help="Directory to save validation reports")
    
    args = parser.parse_args()
    
    print("Starting model validation...")
    print(f"Arguments: {args}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.report_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, args.num_classes, args.backbone, device)
    
    # Load validation data
    print("Loading validation data...")
    val_dataset = ValidationDataset(
        os.path.join(args.metadata_dir, "val_metadata.csv"),
        args.dataset_dir
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Evaluate model
    y_true, y_pred, y_prob = evaluate_model(model, val_loader, device, args.threshold)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Print results
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Average Precision: {metrics['avg_precision']:.4f}")
    print(f"Average Recall: {metrics['avg_recall']:.4f}")
    print(f"Average F1-Score: {metrics['avg_f1']:.4f}")
    print(f"Average AUROC: {metrics['avg_auroc']:.4f}")
    
    # Check target accuracy
    target_min = 0.86
    target_max = 0.92
    
    print("\n" + "="*60)
    print("TARGET ACCURACY CHECK")
    print("="*60)
    accuracy_met = target_min <= metrics['accuracy'] <= target_max
    auroc_met = target_min <= metrics['avg_auroc'] <= target_max
    
    print(f"Accuracy: {metrics['accuracy']:.4f} (Target: {target_min}-{target_max}) {'✅ MET' if accuracy_met else '❌ NOT MET'}")
    print(f"AUROC: {metrics['avg_auroc']:.4f} (Target: {target_min}-{target_max}) {'✅ MET' if auroc_met else '❌ NOT MET'}")
    
    if accuracy_met and auroc_met:
        print("🎉 ALL TARGETS ACHIEVED!")
    else:
        print("⚠️  Some targets not achieved")
    
    # Per-class results
    print("\n" + "="*60)
    print("PER-CLASS RESULTS")
    print("="*60)
    
    per_class_df = pd.DataFrame(metrics['per_class_metrics'])
    print(per_class_df.to_string(index=False, float_format='%.4f'))
    
    # Save results
    results = {
        'model_path': args.model_path,
        'backbone': args.backbone,
        'validation_date': datetime.now().isoformat(),
        'num_samples': len(y_true),
        'overall_metrics': {
            'accuracy': float(metrics['accuracy']),
            'avg_precision': float(metrics['avg_precision']),
            'avg_recall': float(metrics['avg_recall']),
            'avg_f1': float(metrics['avg_f1']),
            'avg_auroc': float(metrics['avg_auroc'])
        },
        'per_class_metrics': metrics['per_class_metrics'],
        'target_achieved': {
            'accuracy': accuracy_met,
            'auroc': auroc_met
        }
    }
    
    # Save JSON results
    results_path = os.path.join(args.report_dir, "validation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV results
    csv_path = os.path.join(args.report_dir, "per_class_metrics.csv")
    per_class_df.to_csv(csv_path, index=False)
    
    print(f"\nValidation completed!")
    print(f"Results saved to: {args.report_dir}")
    print(f"JSON results: {results_path}")
    print(f"CSV results: {csv_path}")

if __name__ == "__main__":
    main()

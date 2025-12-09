# evaluate.py
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import json
from datetime import datetime

# Import custom modules
from utils.model_utils import MultiLabelResNet, TARGET_PATHOLOGIES
from utils.data_loader import ChestXrayDataset
from utils.report_generator import ClassificationReportGenerator

def load_model(model_path, num_classes, backbone="resnext101_32x8d", device="cuda"):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path (str): Path to model checkpoint
        num_classes (int): Number of classes
        backbone (str): Model backbone
        device (str): Device to load model on
        
    Returns:
        torch.nn.Module: Loaded model
    """
    print(f"Loading model from {model_path}")
    
    # Create model
    model = MultiLabelResNet(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
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
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        threshold: Threshold for binary predictions
        
    Returns:
        tuple: (y_true, y_pred, y_prob)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluation"):
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
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

def evaluate_on_splits(model, train_loader, val_loader, test_loader, device, args):
    """
    Evaluate model on train, validation, and test splits.
    
    Args:
        model: Trained model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to run evaluation on
        args: Command line arguments
        
    Returns:
        dict: Results for all splits
    """
    results = {}
    
    # Evaluate on each split
    splits = {
        'train': train_loader,
        'validation': val_loader,
        'test': test_loader
    }
    
    for split_name, loader in splits.items():
        print(f"\nEvaluating on {split_name} set...")
        y_true, y_pred, y_prob = evaluate_model(model, loader, device, args.threshold)
        
        # Calculate basic metrics
        accuracy = (y_pred == y_true).mean()
        
        results[split_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'accuracy': accuracy,
            'num_samples': len(y_true)
        }
        
        print(f"{split_name.capitalize()} Accuracy: {accuracy:.4f}")
        print(f"{split_name.capitalize()} Samples: {len(y_true)}")
    
    return results

def generate_detailed_reports(results, args):
    """
    Generate detailed reports for each split.
    
    Args:
        results (dict): Evaluation results
        args: Command line arguments
    """
    print("\nGenerating detailed reports...")
    
    # Create report generator
    report_generator = ClassificationReportGenerator(output_dir=args.report_dir)
    
    # Generate reports for each split
    for split_name, result in results.items():
        print(f"\nGenerating report for {split_name} set...")
        
        y_true = result['y_true']
        y_pred = result['y_pred']
        y_prob = result['y_prob']
        
        # Generate comprehensive report
        metrics = report_generator.generate_comprehensive_report(
            y_true, y_pred, y_prob,
            model_name=f"{args.backbone}_ChestX-ray_Model",
            dataset_name=f"{split_name.capitalize()} Dataset"
        )
        
        # Save metrics
        metrics_path = os.path.join(args.report_dir, f"{split_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"Report saved for {split_name} set")

def compare_with_targets(results, args):
    """
    Compare results with target accuracy requirements.
    
    Args:
        results (dict): Evaluation results
        args: Command line arguments
    """
    print("\n" + "="*60)
    print("TARGET ACCURACY COMPARISON")
    print("="*60)
    
    target_min = 0.86
    target_max = 0.92
    
    for split_name, result in results.items():
        print(f"\n{split_name.upper()} SET RESULTS:")
        print("-" * 40)
        
        # Calculate AUROC (we'll use a simple approximation for now)
        # In practice, you'd calculate this properly using sklearn
        y_true = result['y_true']
        y_prob = result['y_prob']
        
        # Simple AUROC calculation per class
        auroc_scores = []
        for i in range(len(TARGET_PATHOLOGIES)):
            if np.sum(y_true[:, i]) > 0:  # Only if class exists
                from sklearn.metrics import roc_auc_score
                try:
                    auroc = roc_auc_score(y_true[:, i], y_prob[:, i])
                    auroc_scores.append(auroc)
                except:
                    auroc_scores.append(0.5)  # Default for problematic cases
        
        avg_auroc = np.mean(auroc_scores) if auroc_scores else 0.5
        accuracy = result['accuracy']
        
        print(f"Multi-label Accuracy: {accuracy:.4f}")
        print(f"Average AUROC: {avg_auroc:.4f}")
        
        # Check if targets are met
        accuracy_met = target_min <= accuracy <= target_max
        auroc_met = target_min <= avg_auroc <= target_max
        
        print(f"Accuracy Target ({target_min}-{target_max}): {'✅ MET' if accuracy_met else '❌ NOT MET'}")
        print(f"AUROC Target ({target_min}-{target_max}): {'✅ MET' if auroc_met else '❌ NOT MET'}")
        
        if accuracy_met and auroc_met:
            print("🎉 ALL TARGETS ACHIEVED!")
        else:
            print("⚠️  Some targets not achieved")
    
    print("="*60)

def analyze_per_class_performance(results, args):
    """
    Analyze per-class performance.
    
    Args:
        results (dict): Evaluation results
        args: Command line arguments
    """
    print("\n" + "="*60)
    print("PER-CLASS PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Focus on test set for detailed analysis
    test_result = results['test']
    y_true = test_result['y_true']
    y_pred = test_result['y_pred']
    y_prob = test_result['y_prob']
    
    # Calculate per-class metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    per_class_metrics = []
    
    for i, pathology in enumerate(TARGET_PATHOLOGIES):
        if np.sum(y_true[:, i]) > 0:  # Only if class exists in test set
            precision = precision_score(y_true[:, i], y_pred[:, i])
            recall = recall_score(y_true[:, i], y_pred[:, i])
            f1 = f1_score(y_true[:, i], y_pred[:, i])
            
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
    
    # Create DataFrame and display
    metrics_df = pd.DataFrame(per_class_metrics)
    print("\nPer-Class Metrics:")
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    
    # Save per-class metrics
    metrics_path = os.path.join(args.report_dir, "per_class_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nPer-class metrics saved to: {metrics_path}")
    
    # Identify best and worst performing classes
    print("\n" + "-"*40)
    print("PERFORMANCE SUMMARY:")
    print("-"*40)
    
    best_f1_idx = metrics_df['F1-Score'].idxmax()
    worst_f1_idx = metrics_df['F1-Score'].idxmin()
    
    print(f"Best F1-Score: {metrics_df.iloc[best_f1_idx]['Pathology']} ({metrics_df.iloc[best_f1_idx]['F1-Score']:.4f})")
    print(f"Worst F1-Score: {metrics_df.iloc[worst_f1_idx]['Pathology']} ({metrics_df.iloc[worst_f1_idx]['F1-Score']:.4f})")
    
    best_auroc_idx = metrics_df['AUROC'].idxmax()
    worst_auroc_idx = metrics_df['AUROC'].idxmin()
    
    print(f"Best AUROC: {metrics_df.iloc[best_auroc_idx]['Pathology']} ({metrics_df.iloc[best_auroc_idx]['AUROC']:.4f})")
    print(f"Worst AUROC: {metrics_df.iloc[worst_auroc_idx]['Pathology']} ({metrics_df.iloc[worst_auroc_idx]['AUROC']:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Chest X-ray Classification Model")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--backbone", type=str, default="resnext101_32x8d",
                       choices=["resnet101", "resnext101_32x8d", "efficientnet_b4"],
                       help="CNN backbone architecture")
    parser.add_argument("--num_classes", type=int, default=14,
                       help="Number of pathology classes")
    
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for binary predictions")
    
    # Directories
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="Directory containing processed data")
    parser.add_argument("--report_dir", type=str, default="evaluation_reports",
                       help="Directory to save evaluation reports")
    
    # Evaluation options
    parser.add_argument("--skip_train_eval", action="store_true",
                       help="Skip evaluation on training set")
    parser.add_argument("--detailed_reports", action="store_true",
                       help="Generate detailed reports with visualizations")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.report_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, args.num_classes, args.backbone, device)
    
    # Load data
    print("Loading evaluation data...")
    
    data_loaders = {}
    
    if not args.skip_train_eval:
        train_dataset = ChestXrayDataset(
            os.path.join(args.data_dir, "train_metadata.csv"),
            os.path.join(args.data_dir, "images"),
            is_train=False
        )
        data_loaders['train'] = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
    
    val_dataset = ChestXrayDataset(
        os.path.join(args.data_dir, "val_metadata.csv"),
        os.path.join(args.data_dir, "images"),
        is_train=False
    )
    data_loaders['validation'] = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    test_dataset = ChestXrayDataset(
        os.path.join(args.data_dir, "test_metadata.csv"),
        os.path.join(args.data_dir, "images"),
        is_train=False
    )
    data_loaders['test'] = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    print(f"Loaded data loaders: {list(data_loaders.keys())}")
    
    # Evaluate model
    results = evaluate_on_splits(
        model, 
        train_loader=data_loaders['train'],
        val_loader=data_loaders['validation'], 
        test_loader=data_loaders['test'],
        device=device, 
        args=args
    )
    
    # Generate detailed reports if requested
    if args.detailed_reports:
        generate_detailed_reports(results, args)
    
    # Compare with targets
    compare_with_targets(results, args)
    
    # Analyze per-class performance
    analyze_per_class_performance(results, args)
    
    # Save summary results
    summary = {
        'model_path': args.model_path,
        'backbone': args.backbone,
        'evaluation_date': datetime.now().isoformat(),
        'results': {}
    }
    
    for split_name, result in results.items():
        summary['results'][split_name] = {
            'accuracy': float(result['accuracy']),
            'num_samples': int(result['num_samples'])
        }
    
    summary_path = os.path.join(args.report_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation completed! Summary saved to: {summary_path}")
    print(f"All reports saved to: {args.report_dir}")

if __name__ == "__main__":
    main()

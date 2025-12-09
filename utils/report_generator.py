# utils/report_generator.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.metrics import multilabel_confusion_matrix
import torch
from torchmetrics.classification import (
    MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, 
    MultilabelF1Score, MultilabelAUROC
)
import json
from datetime import datetime

# Import target pathologies
from utils.model_utils import TARGET_PATHOLOGIES

class ClassificationReportGenerator:
    """
    Generates comprehensive classification reports for multi-label chest X-ray classification.
    """
    
    def __init__(self, target_pathologies=None, output_dir="reports"):
        self.target_pathologies = target_pathologies or TARGET_PATHOLOGIES
        self.num_classes = len(self.target_pathologies)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize torchmetrics
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = {
            'accuracy': MultilabelAccuracy(num_labels=self.num_classes, average='micro').to(self.device),
            'precision': MultilabelPrecision(num_labels=self.num_classes, average='micro').to(self.device),
            'recall': MultilabelRecall(num_labels=self.num_classes, average='micro').to(self.device),
            'f1_score': MultilabelF1Score(num_labels=self.num_classes, average='micro').to(self.device),
            'auroc': MultilabelAUROC(num_labels=self.num_classes, average='macro', thresholds=None).to(self.device)
        }
        
        # Per-class metrics
        self.per_class_metrics = {
            'accuracy': MultilabelAccuracy(num_labels=self.num_classes, average='none').to(self.device),
            'precision': MultilabelPrecision(num_labels=self.num_classes, average='none').to(self.device),
            'recall': MultilabelRecall(num_labels=self.num_classes, average='none').to(self.device),
            'f1_score': MultilabelF1Score(num_labels=self.num_classes, average='none').to(self.device),
            'auroc': MultilabelAUROC(num_labels=self.num_classes, average='none', thresholds=None).to(self.device)
        }
    
    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calculate comprehensive metrics for multi-label classification.
        
        Args:
            y_true (np.ndarray): True binary labels (n_samples, n_classes)
            y_pred (np.ndarray): Predicted binary labels (n_samples, n_classes)
            y_prob (np.ndarray): Predicted probabilities (n_samples, n_classes)
            
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        # Convert to torch tensors
        y_true_torch = torch.tensor(y_true, dtype=torch.float32).to(self.device)
        y_pred_torch = torch.tensor(y_pred, dtype=torch.float32).to(self.device)
        
        if y_prob is not None:
            y_prob_torch = torch.tensor(y_prob, dtype=torch.float32).to(self.device)
        else:
            y_prob_torch = y_pred_torch
        
        # Calculate overall metrics
        overall_metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            if metric_name == 'auroc':
                overall_metrics[metric_name] = metric_fn(y_prob_torch, y_true_torch).item()
            else:
                overall_metrics[metric_name] = metric_fn(y_pred_torch, y_true_torch).item()
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for metric_name, metric_fn in self.per_class_metrics.items():
            if metric_name == 'auroc':
                per_class_values = metric_fn(y_prob_torch, y_true_torch).cpu().numpy()
            else:
                per_class_values = metric_fn(y_pred_torch, y_true_torch).cpu().numpy()
            
            per_class_metrics[metric_name] = dict(zip(self.target_pathologies, per_class_values))
        
        # Calculate additional sklearn metrics
        sklearn_metrics = self._calculate_sklearn_metrics(y_true, y_pred, y_prob)
        
        return {
            'overall': overall_metrics,
            'per_class': per_class_metrics,
            'sklearn': sklearn_metrics
        }
    
    def _calculate_sklearn_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate additional metrics using sklearn."""
        metrics = {}
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.target_pathologies, output_dict=True
        )
        
        # Per-class AUROC and Average Precision
        if y_prob is not None:
            metrics['per_class_auroc'] = {}
            metrics['per_class_avg_precision'] = {}
            
            for i, pathology in enumerate(self.target_pathologies):
                if np.sum(y_true[:, i]) > 0:  # Only calculate if class exists in true labels
                    fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
                    metrics['per_class_auroc'][pathology] = auc(fpr, tpr)
                    
                    precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
                    metrics['per_class_avg_precision'][pathology] = average_precision_score(
                        y_true[:, i], y_prob[:, i]
                    )
        
        return metrics
    
    def generate_confusion_matrices(self, y_true, y_pred, save_path=None):
        """
        Generate confusion matrices for multi-label classification.
        
        Args:
            y_true (np.ndarray): True binary labels
            y_pred (np.ndarray): Predicted binary labels
            save_path (str): Path to save the plot
        """
        # Calculate multilabel confusion matrices
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        
        # Create subplots for each class
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, pathology in enumerate(self.target_pathologies):
            cm = mcm[i]
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Predicted No', 'Predicted Yes'],
                       yticklabels=['Actual No', 'Actual Yes'])
            axes[i].set_title(f'{pathology}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(len(self.target_pathologies), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to: {save_path}")
        
        plt.show()
        return fig
    
    def generate_roc_curves(self, y_true, y_prob, save_path=None):
        """
        Generate ROC curves for each class.
        
        Args:
            y_true (np.ndarray): True binary labels
            y_prob (np.ndarray): Predicted probabilities
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        auroc_scores = {}
        
        for i, pathology in enumerate(self.target_pathologies):
            if np.sum(y_true[:, i]) > 0:  # Only plot if class exists
                fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
                auroc = auc(fpr, tpr)
                auroc_scores[pathology] = auroc
                
                axes[i].plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC curve (AUC = {auroc:.3f})')
                axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                axes[i].set_xlim([0.0, 1.0])
                axes[i].set_ylim([0.0, 1.05])
                axes[i].set_xlabel('False Positive Rate')
                axes[i].set_ylabel('True Positive Rate')
                axes[i].set_title(f'{pathology}\nROC Curve')
                axes[i].legend(loc="lower right")
            else:
                axes[i].text(0.5, 0.5, f'{pathology}\nNo positive samples', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{pathology}\nROC Curve')
        
        # Hide unused subplots
        for i in range(len(self.target_pathologies), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {save_path}")
        
        plt.show()
        return fig, auroc_scores
    
    def generate_precision_recall_curves(self, y_true, y_prob, save_path=None):
        """
        Generate Precision-Recall curves for each class.
        
        Args:
            y_true (np.ndarray): True binary labels
            y_prob (np.ndarray): Predicted probabilities
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        avg_precision_scores = {}
        
        for i, pathology in enumerate(self.target_pathologies):
            if np.sum(y_true[:, i]) > 0:  # Only plot if class exists
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
                avg_precision = average_precision_score(y_true[:, i], y_prob[:, i])
                avg_precision_scores[pathology] = avg_precision
                
                axes[i].plot(recall, precision, color='darkorange', lw=2,
                           label=f'PR curve (AP = {avg_precision:.3f})')
                axes[i].set_xlim([0.0, 1.0])
                axes[i].set_ylim([0.0, 1.05])
                axes[i].set_xlabel('Recall')
                axes[i].set_ylabel('Precision')
                axes[i].set_title(f'{pathology}\nPrecision-Recall Curve')
                axes[i].legend(loc="lower left")
            else:
                axes[i].text(0.5, 0.5, f'{pathology}\nNo positive samples',
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{pathology}\nPrecision-Recall Curve')
        
        # Hide unused subplots
        for i in range(len(self.target_pathologies), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curves saved to: {save_path}")
        
        plt.show()
        return fig, avg_precision_scores
    
    def generate_metrics_summary(self, metrics, save_path=None):
        """
        Generate a summary table of all metrics.
        
        Args:
            metrics (dict): Calculated metrics dictionary
            save_path (str): Path to save the CSV file
        """
        # Create summary DataFrame
        summary_data = []
        
        for pathology in self.target_pathologies:
            row = {
                'Pathology': pathology,
                'Accuracy': metrics['per_class']['accuracy'].get(pathology, 0.0),
                'Precision': metrics['per_class']['precision'].get(pathology, 0.0),
                'Recall': metrics['per_class']['recall'].get(pathology, 0.0),
                'F1-Score': metrics['per_class']['f1_score'].get(pathology, 0.0),
                'AUROC': metrics['per_class']['auroc'].get(pathology, 0.0)
            }
            
            # Add sklearn metrics if available
            if 'per_class_auroc' in metrics['sklearn']:
                row['AUROC (sklearn)'] = metrics['sklearn']['per_class_auroc'].get(pathology, 0.0)
            if 'per_class_avg_precision' in metrics['sklearn']:
                row['Avg Precision'] = metrics['sklearn']['per_class_avg_precision'].get(pathology, 0.0)
            
            summary_data.append(row)
        
        # Add overall metrics row
        overall_row = {
            'Pathology': 'OVERALL (Micro-avg)',
            'Accuracy': metrics['overall']['accuracy'],
            'Precision': metrics['overall']['precision'],
            'Recall': metrics['overall']['recall'],
            'F1-Score': metrics['overall']['f1_score'],
            'AUROC': metrics['overall']['auroc']
        }
        summary_data.append(overall_row)
        
        summary_df = pd.DataFrame(summary_data)
        
        if save_path:
            summary_df.to_csv(save_path, index=False)
            print(f"Metrics summary saved to: {save_path}")
        
        return summary_df
    
    def generate_comprehensive_report(self, y_true, y_pred, y_prob=None, 
                                     model_name="ChestX-ray Model", 
                                     dataset_name="Test Dataset"):
        """
        Generate a comprehensive classification report with all visualizations and metrics.
        
        Args:
            y_true (np.ndarray): True binary labels
            y_pred (np.ndarray): Predicted binary labels
            y_prob (np.ndarray): Predicted probabilities
            model_name (str): Name of the model
            dataset_name (str): Name of the dataset
            
        Returns:
            dict: Complete metrics dictionary
        """
        print(f"Generating comprehensive report for {model_name} on {dataset_name}")
        print(f"Dataset shape: {y_true.shape}")
        
        # Calculate all metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        # Create timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate visualizations
        print("Generating confusion matrices...")
        cm_path = os.path.join(self.output_dir, f"confusion_matrices_{timestamp}.png")
        self.generate_confusion_matrices(y_true, y_pred, cm_path)
        
        if y_prob is not None:
            print("Generating ROC curves...")
            roc_path = os.path.join(self.output_dir, f"roc_curves_{timestamp}.png")
            roc_fig, auroc_scores = self.generate_roc_curves(y_true, y_prob, roc_path)
            
            print("Generating Precision-Recall curves...")
            pr_path = os.path.join(self.output_dir, f"precision_recall_curves_{timestamp}.png")
            pr_fig, avg_precision_scores = self.generate_precision_recall_curves(y_true, y_prob, pr_path)
        
        # Generate metrics summary
        print("Generating metrics summary...")
        summary_path = os.path.join(self.output_dir, f"metrics_summary_{timestamp}.csv")
        summary_df = self.generate_metrics_summary(metrics, summary_path)
        
        # Save detailed metrics as JSON
        json_path = os.path.join(self.output_dir, f"detailed_metrics_{timestamp}.json")
        with open(json_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_metrics = self._convert_metrics_for_json(metrics)
            json.dump(json_metrics, f, indent=2)
        
        print(f"Comprehensive report generated!")
        print(f"Files saved to: {self.output_dir}")
        print(f"Timestamp: {timestamp}")
        
        # Print key metrics
        print("\n" + "="*50)
        print("KEY METRICS SUMMARY")
        print("="*50)
        print(f"Multi-label Accuracy: {metrics['overall']['accuracy']:.4f}")
        print(f"Multi-label Precision: {metrics['overall']['precision']:.4f}")
        print(f"Multi-label Recall: {metrics['overall']['recall']:.4f}")
        print(f"Multi-label F1-Score: {metrics['overall']['f1_score']:.4f}")
        print(f"Multi-label AUROC: {metrics['overall']['auroc']:.4f}")
        print("="*50)
        
        return metrics
    
    def _convert_metrics_for_json(self, metrics):
        """Convert metrics dictionary to JSON-serializable format."""
        json_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                json_metrics[key] = self._convert_metrics_for_json(value)
            elif isinstance(value, np.ndarray):
                json_metrics[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_metrics[key] = float(value)
            else:
                json_metrics[key] = value
        
        return json_metrics

def generate_report_from_predictions(y_true, y_pred, y_prob=None, 
                                   output_dir="reports",
                                   model_name="ChestX-ray Model",
                                   dataset_name="Test Dataset"):
    """
    Convenience function to generate a comprehensive report from predictions.
    
    Args:
        y_true (np.ndarray): True binary labels (n_samples, n_classes)
        y_pred (np.ndarray): Predicted binary labels (n_samples, n_classes)
        y_prob (np.ndarray): Predicted probabilities (n_samples, n_classes)
        output_dir (str): Directory to save reports
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Complete metrics dictionary
    """
    generator = ClassificationReportGenerator(output_dir=output_dir)
    return generator.generate_comprehensive_report(
        y_true, y_pred, y_prob, model_name, dataset_name
    )

if __name__ == "__main__":
    # Example usage with dummy data
    print("Testing ClassificationReportGenerator with dummy data...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    n_classes = len(TARGET_PATHOLOGIES)
    
    # Generate random true labels (sparse)
    y_true = np.random.binomial(1, 0.1, (n_samples, n_classes))
    
    # Generate random predictions
    y_prob = np.random.random((n_samples, n_classes))
    y_pred = (y_prob > 0.5).astype(int)
    
    # Generate report
    metrics = generate_report_from_predictions(
        y_true, y_pred, y_prob,
        output_dir="test_reports",
        model_name="Test Model",
        dataset_name="Test Dataset"
    )
    
    print("Test completed successfully!")

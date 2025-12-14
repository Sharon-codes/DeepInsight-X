# utils/model_utils.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast # For mixed precision training
try:
    from torch.utils.tensorboard import SummaryWriter # For logging [19]
except ImportError:
    SummaryWriter = None
from sklearn.metrics import roc_auc_score # For calculating AUROC [19]
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelAUROC # For advanced metrics
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np

# Import custom dataset and preprocessing
from utils.data_loader import get_data_loaders
from utils.preprocessing import TARGET_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
import torchvision.models as models

# Define your 14 target pathologies (must match preprocessing.py)
TARGET_PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

class MultiLabelResNet(nn.Module):
    def __init__(self, num_classes, backbone='resnext101_32x8d', pretrained=True):
        super(MultiLabelResNet, self).__init__()
        # Using various backbones for different performance levels
        if backbone == 'resnet101':
            self.base_model = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnext101_32x8d': # A more powerful ResNeXt variant
            self.base_model = models.resnext101_32x8d(pretrained=pretrained)
        elif backbone == 'efficientnet_b4': # EfficientNet for good balance of accuracy and efficiency
            self.base_model = models.efficientnet_b4(pretrained=pretrained)
        elif backbone == 'convnext_large': # ConvNeXT Large for highest accuracy
            self.base_model = models.convnext_large(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze all parameters in the base model initially
        # This is a common practice for transfer learning to prevent catastrophic forgetting
        # You can unfreeze layers later for fine-tuning
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Replace the final classification layer for multi-label output
        if hasattr(self.base_model, 'fc'):  # ResNet/ResNeXt
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, num_classes)
            )
        elif hasattr(self.base_model, 'classifier'):  # EfficientNet/ConvNeXT
            # For ConvNeXT, we need to handle the classifier differently
            if backbone == 'convnext_large':
                # ConvNeXT has a different structure: Norm -> Flatten -> Linear
                # We only want to replace the final Linear layer (index 2) to match the trained checkpoint
                in_features = self.base_model.classifier[2].in_features
                self.base_model.classifier[2] = nn.Linear(in_features, num_classes)
            else:
                # For EfficientNet
                in_features = self.base_model.classifier[-1].in_features
                self.base_model.classifier = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(in_features, num_classes)
                )
        else:
            raise ValueError(f"Unknown model architecture for backbone: {backbone}")

    def forward(self, x):
        # The BCEWithLogitsLoss combines Sigmoid and Binary Cross Entropy for numerical stability
        # So, the model's forward pass should output logits (raw scores) [1, 22]
        return self.base_model(x)

def setup_distributed_training(rank, world_size):
    """Initialize the process group for distributed training."""
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) # Assign each process to a specific GPU

def cleanup_distributed_training():
    """Destroy the process group after training is complete."""
    dist.destroy_process_group()

def train_model(rank, world_size, args):
    setup_distributed_training(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Initialize model
    model = MultiLabelResNet(num_classes=len(TARGET_PATHOLOGIES), backbone=args.backbone, pretrained=True).to(device)
    ddp_model = DDP(model, device_ids=[rank]) # Wrap model with DDP

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss() # Ideal for multi-label classification [1, 22]
    optimizer = optim.AdamW(ddp_model.parameters(), lr=args.learning_rate, weight_decay=1e-5) # AdamW for better regularization

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', # Monitor validation loss
        factor=0.1, # Reduce LR by 10%
        patience=5, # Number of epochs with no improvement after which learning rate will be reduced
        verbose=True
    )

    # Mixed precision training scaler
    scaler = GradScaler()

    # Data loaders
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.processed_data_dir,
        args.train_metadata_path,
        args.val_metadata_path,
        args.test_metadata_path,
        args.batch_size,
        args.num_workers,
        world_size,
        rank
    )

    # Metrics for evaluation
    metrics = {
        'accuracy': MultilabelAccuracy(num_labels=num_classes, average='micro').to(device),
        'precision': MultilabelPrecision(num_labels=num_classes, average='micro').to(device),
        'recall': MultilabelRecall(num_labels=num_classes, average='micro').to(device),
        'f1_score': MultilabelF1Score(num_labels=num_classes, average='micro').to(device),
        'auroc': MultilabelAUROC(num_labels=num_classes, average='macro', thresholds=None).to(device) # AUROC is good for imbalanced datasets
    }

    # TensorBoard writer (only on rank 0 to avoid redundant logs) [19]
    if rank == 0:
        run_id = os.environ.get('SLURM_JOB_ID', 'local')
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f"run_{run_id}"))
        print(f"TensorBoard logs available at: {os.path.join(args.log_dir, f'run_{run_id}')}")

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch) # Important for shuffling in DDP [19]
        ddp_model.train()
        running_loss = 0.0
        
        # Gradient accumulation
        optimizer.zero_grad()
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training (Rank {rank})")):
            images, labels = images.to(device), labels.to(device)

            with autocast(): # Mixed precision forward pass
                outputs = ddp_model(images)
                loss = criterion(outputs, labels)
            
            # Scale loss and perform backward pass
            scaler.scale(loss).backward()

            # Perform optimizer step only after accumulating gradients for `gradient_accumulation_steps` batches
            if (i + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item()

        # Ensure all gradients are updated before validation if last batch didn't trigger step
        if (i + 1) % args.gradient_accumulation_steps!= 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = running_loss / len(train_loader)

        # Validation phase (only on rank 0 for simplicity) [19]
        if rank == 0:
            ddp_model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation (Rank {rank})"):
                    images, labels = images.to(device), labels.to(device)
                    with autocast():
                        outputs = ddp_model(images)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    all_preds.append(torch.sigmoid(outputs).cpu()) # Apply sigmoid for probability-based metrics
                    all_labels.append(labels.cpu())

            avg_val_loss = val_loss / len(val_loader)
            
            # Concatenate all predictions and labels
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            # Calculate and log metrics
            val_accuracy = metrics['accuracy'](all_preds, all_labels).item()
            val_precision = metrics['precision'](all_preds, all_labels).item()
            val_recall = metrics['recall'](all_preds, all_labels).item()
            val_f1 = metrics['f1_score'](all_preds, all_labels).item()
            val_auroc = metrics['auroc'](all_preds, all_labels).item()

            print(f"Epoch {epoch+1}/{args.epochs} (Rank {rank}):")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}")
            print(f"  Val Precision: {val_precision:.4f}")
            print(f"  Val Recall: {val_recall:.4f}")
            print(f"  Val F1 Score: {val_f1:.4f}")
            print(f"  Val AUROC: {val_auroc:.4f}")

            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('Metrics/val_accuracy', val_accuracy, epoch)
            writer.add_scalar('Metrics/val_precision', val_precision, epoch)
            writer.add_scalar('Metrics/val_recall', val_recall, epoch)
            writer.add_scalar('Metrics/val_f1_score', val_f1, epoch)
            writer.add_scalar('Metrics/val_auroc', val_auroc, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups['lr'], epoch)

            # Step the scheduler based on validation loss
            scheduler.step(avg_val_loss)

            # Save best model checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_save_path = os.path.join(args.model_save_dir, f"best_model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"Best model saved to {model_save_path} with Val Loss: {best_val_loss:.4f}")
            
            # Save latest model checkpoint
            latest_model_save_path = os.path.join(args.model_save_dir, "latest_model.pth")
            torch.save(model.state_dict(), latest_model_save_path)

    if rank == 0:
        writer.close()
    cleanup_distributed_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distributed Chest X-ray Classification Training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers per GPU")
    parser.add_argument("--backbone", type=str, default='resnext101_32x8d', help="CNN backbone architecture (e.g., resnet101, resnext101_32x8d, efficientnet_b4)")
    parser.add_argument("--processed_data_dir", type=str, default="data/processed", help="Directory for processed images")
    parser.add_argument("--train_metadata_path", type=str, default="data/processed/train_metadata.csv", help="Path to training metadata CSV")
    parser.add_argument("--val_metadata_path", type=str, default="data/processed/val_metadata.csv", help="Path to validation metadata CSV")
    parser.add_argument("--test_metadata_path", type=str, default="data/processed/test_metadata.csv", help="Path to test metadata CSV")
    parser.add_argument("--model_save_dir", type=str, default="models", help="Directory to save trained models")
    parser.add_argument("--log_dir", type=str, default="runs", help="Directory for TensorBoard logs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of batches to accumulate gradients over")
    
    args = parser.parse_args()

    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # These environment variables are set by torchrun or srun
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

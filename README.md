# Chest X-Ray AI - Training Pipeline

Complete pipeline for training a multi-label chest X-ray classification model on NIH, OpenI, and ReXGradient datasets.

## Overview

This repository contains the training pipeline for a ConvNeXt Large model achieving high AUROC (>0.90) on chest X-ray pathology detection.

## Features

- 📦 **Automated Dataset Processing**: Combines NIH ChestX-ray14, OpenI, and ReXGradient (MIMIC)
- 🎯 **Multi-Label Classification**: 14 thoracic pathologies
- 🔥 **Focal Loss**: Handles class imbalance effectively
- 📊 **Comprehensive Metrics**: AUROC, F1, Precision, Recall, Hamming Accuracy
- 🚀 **HPC Ready**: Optimized for high-performance computing environments

## Datasets

### Supported Datasets
1. **NIH ChestX-ray14** (~112K images)
   - Download from: https://nihcc.app.box.com/v/ChestXray-NIHCC
   - Place in: `Dataset/images_001/images` through `images_012/images`

2. **OpenI** (~7.5K images)
   - Auto-downloaded during processing
   - Contains Indiana University reports with findings

3. **ReXGradient (MIMIC)** (~160K images)
   - Download using `download_rex_v2.py`
   - Requires Hugging Face token
   - Place in: `Dataset/ReXGradient/`

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <core-repo-url>
cd chest-xray-core

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
# Download ReXGradient (requires HF token)
python download_rex_v2.py --token YOUR_HF_TOKEN

# NIH dataset: download manually and extract to Dataset/
```

### 3. Generate Combined Metadata

```bash
# Processes all datasets into train_metadata.csv
python create_full_dataset.py
```

### 4. Train Model

```bash
# Recommended settings for AUROC > 0.90
python train_v3.py --epochs 30 --batch_size 32 --lr 5e-5
```

## Project Structure

```
Core/
├── create_full_dataset.py    # Dataset processing & merging
├── download_rex_v2.py        # ReXGradient downloader
├── train_v3.py               # Main training script (optimized)
├── requirements.txt          # Python dependencies
├── models/                   # Saved model checkpoints
│   └── best_model_v3.pth
├── data/
│   └── processed/
│       └── train_metadata.csv
└── utils/
    ├── data_loader.py
    ├── model_utils.py
    ├── grad_cam.py
    └── preprocessing.py
```

## Training Configuration

### Recommended Hyperparameters
```python
--epochs 30              # Enough for convergence
--batch_size 32          # Balanced speed/memory
--lr 5e-5               # Stable learning rate
--weight_decay 0.05     # Regularization
```

### Model Architecture
- **Backbone**: ConvNeXt Large (pretrained on ImageNet)
- **Loss Function**: Focal Loss (gamma=2.0)
- **Optimizer**: AdamW
- **Scheduler**: ReduceLROnPlateau

### Data Augmentation
- Horizontal flip
- Rotation (±25°)
- ShiftScaleRotate
- CoarseDropout (Cutout)
- Brightness/Contrast adjustment

## HPC Deployment

```bash
# Automated training on HPC
chmod +x run_hpc_training.sh
./run_hpc_training.sh
```

## Dataset Processing Details

### create_full_dataset.py

Combines all datasets into a single `train_metadata.csv`:

**NIH ChestX-ray14**:
- Parses `Data_Entry_2017.csv`
- Maps 8 disease labels to 14 target pathologies
- Scans `images_*/images` folders

**OpenI**:
- Uses `indiana_projections.csv` and `indiana_reports.csv`
- Extracts labels from radiology report text
- Maps findings to target pathologies

**ReXGradient (MIMIC)**:
- Loads `metadata/train_metadata.json`
- Extracts labels from Findings/Impression text
- Indexes images from `deid_png.part*` folders

**Output**: `data/processed/train_metadata.csv` with columns:
- `Image Index`: Filename
- `Processed Image Path`: Full path
- `Harmonized Labels`: Space-separated binary vector (14 pathologies)
- `Dataset`: Source (NIH/OpenI/ReXGradient)

## Performance Metrics

### Target Performance
- **AUROC (macro)**: > 0.90
- **F1 Score (micro)**: > 0.70
- **Precision (micro)**: > 0.75

### Monitoring During Training
```
Epoch 15/30
Train Loss: 0.1234
Val Loss: 0.1456
Val AUROC: 0.9123
Val F1: 0.7234
Val Precision: 0.7654
Val Recall: 0.7012
✓ Saved new best model
```

## Troubleshooting

### ReXGradient Download Issues
```bash
# Error: 401 Client Error
# → Accept dataset terms at: https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K

# Error: FileNotFoundError: git
# → Already handled in download_rex_v2.py (git not required)
```

### Out of Memory
```bash
# Reduce batch size
python train_v3.py --batch_size 16
```

### Low AUROC
- Ensure all datasets are processed correctly
- Verify `train_metadata.csv` has 230K+ rows
- Check label distribution (some pathologies are rare)

## Output Files

- `models/best_model_v3.pth`: Best model by validation AUROC
- `data/processed/train_metadata.csv`: Combined dataset metadata
- Training logs: Printed to console

## Testing

```bash
# Verify dataset processing
python verify_metadata.py

# Check data loader
python -c "from utils.data_loader import *; print('OK')"
```

## Related Repository

- **Web Interface**: See the Website repository for deployment and inference

## Contributors

- Nikita Lotlikar - Research & ML
- Sharon Melhi - Research & ML

## Citation

If you use this code, please cite:
- NIH ChestX-ray14: https://arxiv.org/abs/1705.02315
- ReXGradient: https://arxiv.org/abs/2310.01551

## License

For educational and research purposes only.

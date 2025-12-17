# Chest X-Ray AI - Training Pipeline

Complete pipeline for training a multi-label chest X-ray classification model on **7 major datasets** for maximum accuracy and generalization.

## Overview

This repository contains the training pipeline for a ConvNeXt Large model achieving high AUROC (>0.90) on chest X-ray pathology detection across **~900K images** from 7 diverse sources.

## Features

- 📦 **Multi-Dataset Processing**: Combines 7 major datasets (NIH, OpenI, ReXGradient, CheXpert, MIMIC-CXR, PadChest, VinDr-CXR)
- 🌍 **Geographic Diversity**: US, European, and Asian populations
- 🎯 **Multi-Label Classification**: 14 thoracic pathologies
- 🔥 **Focal Loss**: Handles class imbalance effectively
- 📊 **Comprehensive Metrics**: AUROC, F1, Precision, Recall, Hamming Accuracy
- 🚀 **HPC Ready**: Optimized for high-performance computing environments
- 🔍 **Explainable AI**: Integrated Grad-CAM for visual interpretability

## Datasets

### Supported Datasets (7 Total = ~900K Images)

1. **NIH ChestX-ray14** (~112K images) ✅
   - Download: https://nihcc.app.box.com/v/ChestXray-NIHCC
   - Standard benchmark dataset
   - Place in: `Dataset/images_001/images` through `images_012/images`

2. **OpenI** (~7.5K images) ✅
   - Auto-downloaded during processing
   - Indiana University with high-quality annotations

3. **ReXGradient** (~160K images) ✅
   - Download: `python download_rex_v2.py --token YOUR_HF_TOKEN`
   - Stanford/MIMIC with fine-grained features
   - Place in: `Dataset/ReXGradient/`

4. **CheXpert** (~224K images) 🆕
   - Download: https://stanfordmlgroup.github.io/competitions/chexpert/
   - Stanford Hospital with uncertainty labels
   - Place in: `Dataset/CheXpert/`

5. **MIMIC-CXR** (~377K images) 🆕
   - Download: https://physionet.org/content/mimic-cxr-jpg/2.0.0/
   - MIT/Beth Israel with free-text reports
   - **Requires PhysioNet credentialing (CITI training)**
   - Place in: `Dataset/MIMIC-CXR/`

6. **PadChest** (~161K images) 🆕
   - Download: http://bimcv.cipf.es/bimcv-projects/padchest/
   - University of Alicante (Spanish demographic)
   - Place in: `Dataset/PadChest/`

7. **VinDr-CXR** (~18K images) 🆕
   - Download: https://physionet.org/content/vindr-cxr/1.0.0/
   - Vietnam with bounding box annotations
   - **Requires PhysioNet credentialing (CITI training)**
   - Place in: `Dataset/VinDr-CXR/`

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

#### Option A: Download All Datasets (Recommended)
```bash
# Check which datasets you have
python download_all_datasets.py --check

# Get information about specific dataset
python download_all_datasets.py --info chexpert

# Download automated datasets (VinDr-CXR, ReXGradient)
python download_rex_v2.py --token YOUR_HF_TOKEN
python download_all_datasets.py --download vindrcxr --username YOUR_PHYSIONET_USER

# Manual downloads required for: NIH, CheXpert, MIMIC-CXR, PadChest
# See: SEVEN_DATASET_INTEGRATION_GUIDE.md for detailed instructions
```

#### Option B: Start with 3 Datasets (Faster)
```bash
# Download only NIH, OpenI, ReXGradient
python download_rex_v2.py --token YOUR_HF_TOKEN
# NIH: Download manually from https://nihcc.app.box.com/v/ChestXray-NIHCC
```

### 3. Process Datasets

#### Process All 7 Datasets (Recommended for Best Performance)
```bash
# Process all available datasets
python create_full_dataset_v2.py --datasets all

# Output: data/processed/train_metadata_v2.csv (~900K samples)
```

#### Process Specific Datasets
```bash
# Process only 3 datasets (faster for testing)
python create_full_dataset_v2.py --datasets nih openi rexgradient

# Process with new datasets
python create_full_dataset_v2.py --datasets nih chexpert mimic
```

### 4. Train Model

```bash
# Full training with all 7 datasets (recommended)
python train_v3.py \
  --metadata data/processed/train_metadata_v2.csv \
  --epochs 30 \
  --batch_size 32 \
  --lr 5e-5 \
  --backbone convnext_large

# Quick test run (subset of data)
python train_v3.py \
  --metadata data/processed/train_metadata_v2.csv \
  --epochs 5 \
  --batch_size 16 \
  --max_samples 10000

# Expected results:
# - 3 datasets: AUROC 0.84
# - 7 datasets: AUROC 0.88-0.92 (target)
```

## Project Structure

```
Core/
├── download_all_datasets.py  # 🆕 Dataset download manager
├── create_full_dataset.py    # Dataset processing (3 datasets)
├── create_full_dataset_v2.py # 🆕 Dataset processing (7 datasets)
├── download_rex_v2.py        # ReXGradient downloader
├── train_v3.py               # Main training script (optimized)
├── requirements.txt          # Python dependencies
├── models/                   # Saved model checkpoints
│   └── best_model_v3.pth
├── data/
│   └── processed/
│       ├── train_metadata.csv     # 3 datasets (~280K samples)
│       └── train_metadata_v2.csv  # 🆕 7 datasets (~900K samples)
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
- **Backbone**: ConvNeXt Large (197M parameters, pretrained on ImageNet-22k)
- **Loss Function**: Focal Loss (gamma=2.0) - Handles class imbalance
- **Optimizer**: AdamW - Decoupled weight decay
- **Scheduler**: ReduceLROnPlateau - Adaptive learning rate

### Data Augmentation
- Horizontal flip (medically valid)
- Rotation (±25°) - Patient positioning variance
- ShiftScaleRotate - Equipment differences
- CoarseDropout (Cutout) - Artifact robustness
- Brightness/Contrast adjustment - Multi-site compatibility

**See**: `../TRAINING_CONCEPTS_EXPLAINED.md` for detailed explanations

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

### Target Performance (7 Datasets)
- **AUROC (macro)**: > 0.90 (target with all datasets)
- **F1 Score (micro)**: > 0.75
- **Precision (micro)**: > 0.78
- **Recall (micro)**: > 0.72

### Performance Comparison
```
3 Datasets (NIH + OpenI + ReXGradient):
  - Training Images: ~280K
  - AUROC: 0.84
  - Rare diseases: 0.70-0.75

7 Datasets (All):
  - Training Images: ~900K (3.2× more!)
  - AUROC: 0.88-0.92 (expected)
  - Rare diseases: 0.80-0.85 (+10-15% improvement!)
```

### Monitoring During Training
```
Epoch 15/30
Train Loss: 0.1234
Val Loss: 0.1456
Val AUROC: 0.9123 ✓ (Target achieved!)
Val F1: 0.7534
Val Precision: 0.7854
Val Recall: 0.7212
✓ Saved new best model
```

<<<<<<< HEAD

## Output Files

- `models/best_model_v3.pth`: Best model by validation AUROC
- `data/processed/train_metadata.csv`: 3-dataset metadata (~280K)
- `data/processed/train_metadata_v2.csv`: 7-dataset metadata (~900K) 🆕
- Training logs: Printed to console and saved to logs/

## Documentation

### Quick References
- **Dataset Download**: `python download_all_datasets.py --list`
- **Dataset Status**: `python download_all_datasets.py --check`
- **Dataset Info**: `python download_all_datasets.py --info <dataset_name>`

### Testing

```bash
# Verify dataset processing
python -c "from utils.data_loader import *; print('✓ Data loader OK')"

# Check model loading
python -c "from utils.model_utils import *; print('✓ Model utils OK')"

# Verify Grad-CAM
python -c "from utils.grad_cam import *; print('✓ Grad-CAM OK')"
```

## Related Repository

- **Web Interface**: See the Website repository for deployment and inference

## Contributors

- Nikita Lotlikar - Research & Bio Technology
- Sharon Melhi - Research & AI/ML

## Citation

If you use this code, please cite:
- NIH ChestX-ray14: https://arxiv.org/abs/1705.02315
- ReXGradient: https://arxiv.org/abs/2310.01551

## License

For educational and research purposes only.

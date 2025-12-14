
# Instructions for HPC Deployment

**Objective**: Train a ConvNeXt model on NIH + OpenI + ReXGraident datasets to achieve AUROC > 0.90.

## 1. File Transfer List
Transfer the following files from your local `D:/IIT/Core` to your HPC workspace (e.g., `~/project/Core`):

1.  `train_v3.py` (The main training script)
2.  `create_full_dataset.py` (The dataset generation script with OpenI + ReX support)
3.  `utils/` (The entire `utils` folder, containing `grad_cam.py`, `model_utils.py`)
4.  `requirements.txt` (If you have one, or ensure env matches)

## 2. Directory Structure on HPC
Ensure your data on the HPC is organized roughly like this (or update paths in scripts):

```text
~/project/
├── Core/
│   ├── train_v3.py
│   ├── create_full_dataset.py
│   └── utils/
│       └── ...
└── Dataset/
    ├── Data_Entry_2017.csv
    ├── indiana_reports.csv
    ├── indiana_projections.csv
    ├── images/
    │   ├── images_normalized/ (OpenI images)
    │   └── ... (NIH image folders like images_001, etc.)
    └── ReXGradient/
        ├── metadata.csv (or similar)
        └── ... (ReX images)
```

## 3. Running the Training

### Step 1: Generate the Metadata
First, you need to generate the combined `train_metadata.csv`.
Run this command on the HPC:

```bash
python create_full_dataset.py
```
*Note: You might need to edit `DATASET_DIR` in `create_full_dataset.py` (Line ~285) to point to your HPC dataset folder path.*

### Step 2: Start Training
Run the training script. We use `convnext_large` and `ReduceLROnPlateau`.

```bash
python train_v3.py --epochs 30 --lr 5e-5 --batch_size 32
```
*Note: Adjust `--batch_size` based on your HPC GPU memory (e.g., 64 or 16 if needed).*

## 4. Expected Output
- The script will validate every epoch and print `Val AUROC`.
- It will save `best_model_v3.pth` in `Core/models/` when a new high score is reached.
- Monitor `Val Label Accuracy` and `Val AUROC`. The goal is >0.90 AUROC.

## 5. Troubleshooting
- **Missing Images**: If `create_full_dataset.py` reports 0 ReX samples, check the CSV filename in `load_rex_data` function (currently checks for `train.csv`, `metadata.csv`, `cxr_df.csv`).
- **Memory Errors**: Reduce `--batch_size`.
- **Low AUROC**: If it stays ~0.50, check if labels are being parsed correctly (look at the first few rows of `train_metadata.csv`).

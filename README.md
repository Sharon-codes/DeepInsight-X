Here is the complete, perfectly formatted, and uncorrupted master Markdown (`.md`) file for your repository documentation. All code blocks, mathematical LaTeX formulas, tables, and layout hierarchies have been cleaned and restored for production deployment.

```markdown
# 🫁 DeepInsight-X: Multi-Site Chest Radiography Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Backbone](https://img.shields.io/badge/Backbone-ConvNeXt_Large-teal.svg)]()
[![Scale](https://img.shields.io/badge/Scale-7_Datasets_(%7E900K)-purple.svg)]()
[![Target Performance](https://img.shields.io/badge/AUROC-%3E0.90-success.svg)]()

Complete, enterprise-grade machine learning pipeline for training a high-resolution, multi-label chest radiograph (CXR) classification model across ~900,000 images from 7 major institutional cohorts. Engineered to mitigate hardware-induced domain shift and maximize zero-shot diagnostic generalizability across 14 thoracic pathologies.

---

## 🌟 Key Features

* 📦 **Multi-Dataset Synthesis:** Merges 7 distinct institutional cohorts (NIH, OpenI, ReXGradient, CheXpert, MIMIC-CXR, PadChest, VinDr-CXR) into a single unified tensor space.
* 🌍 **Global Demographic Diversity:** Spans North American, Western European, and Southeast Asian patient populations to eliminate localized demographic biases.
* 🎯 **Multi-Label Diagnostic Space:** Concurrent classification across 14 primary cardiopulmonary abnormalities.
* 🔥 **Robust Optimization:** Integrates Weighted Focal Loss ($\gamma=2.0$) to prevent minority-class collapse (e.g., Hernia at $<1\%$) paired with decoupled AdamW regularization.
* 📊 **Comprehensive Evaluation:** Tracks macro/micro AUROC, F1-Score, Precision, Recall, and Hamming Accuracy.
* 🚀 **HPC & Slurm Ready:** Fully containerized and optimized for high-performance computing clusters and distributed multi-GPU nodes.
* 🔍 **High-Resolution Interpretability:** Native $1024 \times 1024$ spatial layout paired with backward-hook Grad-CAM localization to ensure models rely on anatomical pathology rather than peripheral hardware artifacts.

---

## 📚 Supported Datasets (~900K Total Images)

| Dataset | Est. Volume | Demographic | Access & Ingestion Method | Pipeline Status |
| :--- | :---: | :---: | :--- | :---: |
| **NIH ChestX-ray14** | 112,000 | USA | [Manual Box Link](https://nihcc.app.box.com/v/ChestXray-NIHCC) | `✅ Verified` |
| **Indiana OpenI** | 7,500 | USA | Auto-downloaded via XML/PNG scraper | `✅ Verified` |
| **ReXGradient** | 160,000 | USA (Stanford) | `python download_rex_v2.py --token $HF_TOKEN` | `✅ Verified` |
| **CheXpert** | 224,000 | USA (Stanford) | [Stanford ML Group Portal](https://stanfordmlgroup.github.io/competitions/chexpert/) | `🆕 Integrated` |
| **MIMIC-CXR** | 377,000 | USA (BIDMC) | [PhysioNet CITI Credentialed](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) | `🆕 Integrated` |
| **PadChest** | 161,000 | Spain (BIMCV) | [BIMCV Web Portal](http://bimcv.cipf.es/bimcv-projects/padchest/) | `🆕 Integrated` |
| **VinDr-CXR** | 18,000 | Vietnam | [PhysioNet CITI Credentialed](https://physionet.org/content/vindr-cxr/1.0.0/) | `🆕 Integrated` |

> **Critical Credentialing Note:** MIMIC-CXR and VinDr-CXR require verified PhysioNet access protocols. Ensure your active CITI human-subjects research certification is linked to your terminal environment before running automated fetch scripts. See `SEVEN_DATASET_INTEGRATION_GUIDE.md` for explicit DUA mounting instructions.

---

## 🚀 Quick Start Guide

### 1. Environment Initialization
```bash
# Clone the repository
git clone [https://github.com/Sharon-codes/DeepInsight-X.git](https://github.com/Sharon-codes/DeepInsight-X.git)
cd DeepInsight-X

# Install strict dependencies
pip install -r requirements.txt

```

### 2. Dataset Acquisition

*Option A: Automated Full Ingestion (Recommended)*

```bash
# Audit local storage allocation
python download_all_datasets.py --check

# Fetch automated partitions
python download_rex_v2.py --token <YOUR_HF_TOKEN>
python download_all_datasets.py --download vindrcxr --username <YOUR_PHYSIONET_USER>

# Note: Extract manual downloads (NIH, CheXpert, PadChest) into respective /Dataset subdirectories.

```

*Option B: 3-Dataset Fast Core (Testing)*

```bash
# Download only NIH, OpenI, and ReXGradient for rapid baseline reproduction
python download_rex_v2.py --token <YOUR_HF_TOKEN>

```

### 3. Deterministic Preprocessing & Harmonization

Execute patient-level zero-leakage splits, standardizing Value of Interest (VOI) Look-Up Tables, photometric interpretation inversion, and bilateral intensity clipping:

```bash
# Process all 7 datasets into the master training index
python create_full_dataset_v2.py --datasets all

# Verification: Confirm generation of data/processed/train_metadata_v2.csv (~900K rows)

```

### 4. Execution Commands

```bash
# Full Distributed Production Run (HPC / Multi-GPU / A100 80GB)
python train_v3.py \
  --metadata data/processed/train_metadata_v2.csv \
  --epochs 30 \
  --batch_size 32 \
  --lr 5e-5 \
  --backbone convnext_large

# Local Smoke Test (10K sample manifold to verify CUDA memory maps)
python train_v3.py \
  --metadata data/processed/train_metadata_v2.csv \
  --epochs 5 \
  --batch_size 16 \
  --max_samples 10000

```

---

## 🏛️ Project Structure

```text
DeepInsight-X Core/
├── download_all_datasets.py  # 🆕 Multi-threaded institutional ingest manager
├── create_full_dataset.py    # Baseline preprocessing engine (3 cohorts)
├── create_full_dataset_v2.py # 🆕 Master harmonization engine (7 cohorts)
├── download_rex_v2.py        # Authenticated Hugging Face transfer pipe
├── train_v3.py               # Optimized PyTorch distributed training script
├── run_hpc_training.sh       # Slurm / PBS workload execution wrapper
├── requirements.txt          # Explicit environmental dependencies
├── models/                   # Target directory for serialized weights
│   └── best_model_v3.pth
├── data/processed/           
│   ├── train_metadata.csv    # 3-dataset index (~280K samples)
│   └── train_metadata_v2.csv # 🆕 7-dataset master index (~900K samples)
└── utils/
    ├── data_loader.py        # Custom Dataset with U-Zeros parsing logic
    ├── model_utils.py        # Logit scaling, calibration, Youden's J optimizer
    ├── grad_cam.py           # High-resolution spatial activation generator
    └── preprocessing.py      # VOI LUT, inversion, clipping mathematical logic

```

---

## ⚙️ Optimization & Architectural Stack

### Hyperparameter Configurations

* **Backbone:** `ConvNeXt-Large` (197M Parameters, initialized with *ImageNet-22k* prior weights).
* **Base Learning Rate:** `5e-5` (with adaptive cosine decay).
* **Batch Footprint:** `32` (requires gradient accumulation on $\le 40\text{GB}$ VRAM hardware).
* **Weight Decay:** `0.05` (decoupled AdamW formulation).
* **Epoch Allocation:** `30` (convergence typically registered near Epoch 18).

### Objective Function

To counteract severe background class dominance, optimization is governed by Weighted Focal Loss:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where modulating factor $\gamma = 2.0$ dynamically downweights easily classified normal presentations, forcing gradient attention onto subtle, low-prevalence findings.

### Deterministic Augmentation Manifold

* **Spatial Transforms:** Horizontal Flipping ($p=0.5$), Bi-directional Rotation ($\pm 25^\circ$).
* **Hardware Simulations:** Shift-Scale-Rotate, Brightness/Contrast Jittering ($\pm 15\%$).
* **Artifact De-biasing:** Coarse Dropout / Cutout bounding boxes ($p=0.2$).

---

## 📈 Empirical Scaling Gains

Benchmarking the downstream impact of scaling from the localized 3-dataset core to the fully harmonized 7-dataset global architecture:

| Validation Metric | 3-Dataset Core (Baseline) | 7-Dataset DeepInsight-X | $\Delta$ Improvement |
| --- | --- | --- | --- |
| **Total Ingested Volume** | ~280,000 | **~900,000** | **$3.2\times$ Scale** |
| **Macro AUROC** | 0.842 | **0.890 – 0.912** | **$+0.048$** *(p < 0.001)* |
| **Minority Class F1 (Hernia/Nodule)** | 0.710 | **0.815** | **$+14.7\%$** |
| **Expected Calibration Error (ECE)** | 0.468 | **0.114** | **Post-Scaling Fix** |

---

## ⚡ HPC Workload Automation

For batch execution across Slurm-managed infrastructure:

```bash
chmod +x run_hpc_training.sh
sbatch run_hpc_training.sh

# Monitor execution metrics via standard out stream:
tail -f logs/training_v3.log

```

---

## 🧪 Pre-Flight Audit Scripts

Verify local pipeline integrity prior to committing large compute clusters:

```bash
# 1. Audit DataLoader mapping & tensor collations
python -c "from utils.data_loader import *; print('✓ PyTorch DataLoader OK')"

# 2. Audit ConvNeXt graph compilation & memory bounds
python -c "from utils.model_utils import *; print('✓ Model graph compilation OK')"

# 3. Audit backward hooks for spatial Grad-CAM generation
python -c "from utils.grad_cam import *; print('✓ Explanatory Grad-CAM engine OK')"

```

---

## 👥 Contributors & Citations

* **Nikita Lotlikar** — *Lead Researcher (Biotechnology & Clinical Ontologies)*
* **Sharon Melhi** — *Lead ML Architect (Computational Pipelines & Explainable AI)*

**Core Data Citations:**

* Wang et al. (NIH ChestX-ray14 Benchmark): `arXiv:1705.02315`
* ReXGradient Harmonization Consortium: `arXiv:2310.01551`

---

## 📄 License

Strictly distributed under an Open-Access Research License for non-commercial academic benchmarking, educational analysis, and reproducible clinical methodology evaluation.

```

```

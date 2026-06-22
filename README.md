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

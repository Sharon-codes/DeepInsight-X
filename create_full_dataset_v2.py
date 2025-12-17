"""
Enhanced Dataset Processor for 7 Chest X-Ray Datasets
Processes: NIH, OpenI, ReXGradient, CheXpert, MIMIC-CXR, PadChest, VinDr-CXR

Usage:
    python create_full_dataset_v2.py --datasets all
    python create_full_dataset_v2.py --datasets nih chexpert mimic
    python create_full_dataset_v2.py --list
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os
import json
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Target pathologies (14 classes)
TARGET_PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

OUTPUT_DIR = "data/processed"

# Label harmonization mapping
LABEL_MAPPING = {
    'atelectasis': 'Atelectasis',
    'cardiomegaly': 'Cardiomegaly',
    'enlarged cardiomediastinum': 'Cardiomegaly',
    'effusion': 'Effusion',
    'pleural effusion': 'Effusion',
    'infiltration': 'Infiltration',
    'infiltrate': 'Infiltration',
    'mass': 'Mass',
    'nodule': 'Nodule',
    'pneumonia': 'Pneumonia',
    'pneumothorax': 'Pneumothorax',
    'consolidation': 'Consolidation',
    'edema': 'Edema',
    'pulmonary edema': 'Edema',
    'emphysema': 'Emphysema',
    'fibrosis': 'Fibrosis',
    'thickening': 'Pleural_Thickening',
    'pleural thickening': 'Pleural_Thickening',
    'hernia': 'Hernia',
    'hiatal hernia': 'Hernia',
    'lung opacity': 'Infiltration',
    'opacity': 'Infiltration',
}


def get_image_paths(directory, extensions=['.png', '.jpg', '.jpeg', '.dcm']):
    """Recursively index all images in directory"""
    image_map = {}
    count = 0
    
    for root, dirs, files in os.walk(directory):
        for f in files:
            if any(f.lower().endswith(ext) for ext in extensions):
                image_map[f] = os.path.join(root, f)
                count += 1
                if count % 10000 == 0:
                    print(f"  Indexed {count} images...", end='\r')
    
    print(f"  Indexed {count} total images in {directory}")
    return image_map


def harmonize_labels_from_text(text):
    """Extract labels from free text (reports, findings)"""
    if not text or str(text).lower() == 'nan':
        return np.zeros(len(TARGET_PATHOLOGIES), dtype=int)
    
    text = str(text).lower()
    label_vec = np.zeros(len(TARGET_PATHOLOGIES), dtype=int)
    
    for key, target in LABEL_MAPPING.items():
        if key in text and target in TARGET_PATHOLOGIES:
            idx = TARGET_PATHOLOGIES.index(target)
            label_vec[idx] = 1
    
    return label_vec


# ==================== DATASET PROCESSORS ====================

def process_nih(dataset_dir):
    """Process NIH ChestX-ray14 dataset"""
    logger.info("Processing NIH ChestX-ray14...")
    
    csv_path = os.path.join(dataset_dir, "Data_Entry_2017.csv")
    if not os.path.exists(csv_path):
        logger.warning("NIH Data_Entry_2017.csv not found. Skipping.")
        return[]
    
    df = pd.read_csv(csv_path)
    image_map = get_image_paths(dataset_dir)
    
    data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="NIH"):
        img_name = row['Image Index']
        if img_name not in image_map:
            continue
        
        raw_labels = row['Finding Labels'].split('|')
        label_vec = np.zeros(len(TARGET_PATHOLOGIES), dtype=int)
        
        for label in raw_labels:
            if label in TARGET_PATHOLOGIES:
                idx = TARGET_PATHOLOGIES.index(label)
                label_vec[idx] = 1
        
        data.append({
            'Image Index': img_name,
            'Processed Image Path': image_map[img_name],
            'Harmonized Labels': ' '.join(map(str, label_vec)),
            'Dataset': 'NIH',
            'Split': 'train'  # Will be split later
        })
    
    logger.info(f"Loaded {len(data)} NIH samples")
    return data


def process_openi(dataset_dir):
    """Process OpenI (Indiana University) dataset"""
    logger.info("Processing OpenI...")
    
    reports_path = os.path.join(dataset_dir, 'indiana_reports.csv')
    projections_path = os.path.join(dataset_dir, 'indiana_projections.csv')
    
    if not os.path.exists(reports_path):
        logger.warning("OpenI files not found. Skipping.")
        return []
    
    reports = pd.read_csv(reports_path)
    projections = pd.read_csv(projections_path)
    merged = pd.merge(reports, projections, on='uid', how='inner')
    
    image_map = get_image_paths(os.path.join(dataset_dir, 'images'))
    
    data = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="OpenI"):
        img_name = row['filename']
        if img_name not in image_map:
            continue
        
        # Extract labels from  MeSH and Problems
        text = f"{row.get('MeSH', '')} {row.get('Problems', '')}"
        label_vec = harmonize_labels_from_text(text)
        
        data.append({
            'Image Index': img_name,
            'Processed Image Path': image_map[img_name],
            'Harmonized Labels': ' '.join(map(str, label_vec)),
            'Dataset': 'OpenI',
            'Split': 'train'
        })
    
    logger.info(f"Loaded {len(data)} OpenI samples")
    return data


def process_chexpert(dataset_dir):
    """Process CheXpert (Stanford) dataset"""
    logger.info("Processing CheXpert...")
    
    chexpert_dir = os.path.join(dataset_dir, 'CheXpert')
    train_csv = os.path.join(chexpert_dir, 'train.csv')
    valid_csv = os.path.join(chexpert_dir, 'valid.csv')
    
    if not os.path.exists(train_csv):
        logger.warning("CheXpert not found. Skipping.")
        logger.info("Download from: https://stanfordmlgroup.github.io/competitions/chexpert/")
        return []
    
    # CheXpert has uncertainty labels (0, 1, -1)
    # -1 means uncertain, we'll treat as 0 for now
    # You can also try U-ignore, U-ones, U-zeros strategies
    
    data = []
    
    for csv_file, split_name in [(train_csv, 'train'), (valid_csv, 'valid')]:
        if not os.path.exists(csv_file):
            continue
        
        df = pd.read_csv(csv_file)
        
        # CheXpert columns (they have 14 observations including our 14)
        chexpert_cols = {
            'Atelectasis': 'Atelectasis',
            'Cardiomegaly': 'Cardiomegaly',
            'Consolidation': 'Consolidation',
            'Edema': 'Edema',
            'Pleural Effusion': 'Effusion',
            'Pneumonia': 'Pneumonia',
            'Pneumothorax': 'Pneumothorax',
            'Lung Opacity': 'Infiltration',  # Map to Infiltration
            'Enlarged Cardiomediastinum': 'Cardiomegaly',  # Also cardiomegaly
        }
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"CheXpert-{split_name}"):
            # CheXpert uses relative paths
            img_path = os.path.join(chexpert_dir, row['Path'])
            if not os.path.exists(img_path):
                continue
            
            label_vec = np.zeros(len(TARGET_PATHOLOGIES), dtype=int)
            
            for chex_col, target_col in chexpert_cols.items():
                if chex_col in df.columns and target_col in TARGET_PATHOLOGIES:
                    val = row[chex_col]
                    # CheXpert uses: 1=positive, 0=negative, -1=uncertain, NaN=not mentioned
                    if val == 1.0:  # Only use definite positives
                        idx = TARGET_PATHOLOGIES.index(target_col)
                        label_vec[idx] = 1
            
            data.append({
                'Image Index': os.path.basename(row['Path']),
                'Processed Image Path': img_path,
                'Harmonized Labels': ' '.join(map(str, label_vec)),
                'Dataset': 'CheXpert',
                'Split': split_name
            })
    
    logger.info(f"Loaded {len(data)} CheXpert samples")
    return data


def process_mimic(dataset_dir):
    """Process MIMIC-CXR (MIT/Beth Israel) dataset"""
    logger.info("Processing MIMIC-CXR...")
    
    mimic_dir = os.path.join(dataset_dir, 'MIMIC-CXR')
    metadata_csv = os.path.join(mimic_dir, 'mimic-cxr-2.0.0-metadata.csv')
    chexpert_csv = os.path.join(mimic_dir, 'mimic-cxr-2.0.0-chexpert.csv')
    split_csv = os.path.join(mimic_dir, 'mimic-cxr-2.0.0-split.csv')
    
    if not os.path.exists(metadata_csv):
        logger.warning("MIMIC-CXR not found. Skipping.")
        logger.info("Download from: https://physionet.org/content/mimic-cxr-jpg/2.0.0/")
        logger.info("Requires PhysioNet credentialing (CITI training)")
        return []
    
    metadata = pd.read_csv(metadata_csv)
    chexpert = pd.read_csv(chexpert_csv)
    splits = pd.read_csv(split_csv)
    
    # Merge all
    df = metadata.merge(chexpert, on=['subject_id', 'study_id'], how='inner')
    df = df.merge(splits, on=['subject_id', 'study_id', 'dicom_id'], how='left')
    
    # MIMIC has folder structure: files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
    data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="MIMIC-CXR"):
        # Construct path
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        dicom_id = str(row['dicom_id'])
        
        # Path pattern: p10/p10000032/s50414267/...
        p_folder = f"p{subject_id[:2]}"
        subject_folder = f"p{subject_id}"
        study_folder = f"s{study_id}"
        
        img_path = os.path.join(mimic_dir, 'files', p_folder, subject_folder, study_folder, f"{dicom_id}.jpg")
        
        if not os.path.exists(img_path):
            continue
        
        # Use CheXpert labels (MIMIC provides CheXpert-labeled data)
        label_vec = np.zeros(len(TARGET_PATHOLOGIES), dtype=int)
        
        mimic_to_target = {
            'Atelectasis': 'Atelectasis',
            'Cardiomegaly': 'Cardiomegaly',
            'Consolidation': 'Consolidation',
            'Edema': 'Edema',
            'Pleural Effusion': 'Effusion',
            'Pneumonia': 'Pneumonia',
            'Pneumothorax': 'Pneumothorax',
        }
        
        for mimic_col, target_col in mimic_to_target.items():
            if mimic_col in df.columns and target_col in TARGET_PATHOLOGIES:
                val = row[mimic_col]
                if val == 1.0:
                    idx = TARGET_PATHOLOGIES.index(target_col)
                    label_vec[idx] = 1
        
        data.append({
            'Image Index': f"{dicom_id}.jpg",
            'Processed Image Path': img_path,
            'Harmonized Labels': ' '.join(map(str, label_vec)),
            'Dataset': 'MIMIC-CXR',
            'Split': row.get('split', 'train')
        })
    
    logger.info(f"Loaded {len(data)} MIMIC-CXR samples")
    return data


def process_padchest(dataset_dir):
    """Process PadChest (University of Alicante) dataset"""
    logger.info("Processing PadChest...")
    
    padchest_dir = os.path.join(dataset_dir, 'PadChest')
    csv_path = os.path.join(padchest_dir, 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
    
    if not os.path.exists(csv_path):
        logger.warning("PadChest not found. Skipping.")
        logger.info("Download from: http://bimcv.cipf.es/bimcv-projects/padchest/")
        return []
    
    df = pd.read_csv(csv_path)
    
    # PadChest has Spanish labels - need translation
    padchest_mapping = {
        'atelectasis': 'Atelectasis',
        'cardiomegaly': 'Cardiomegaly',
        'consolidation': 'Consolidation',
        'edema': 'Edema',
        'pleural effusion': 'Effusion',
        'pneumonia': 'Pneumonia',
        'pneumothorax': 'Pneumothorax',
        'emphysema': 'Emphysema',
        'fibrosis': 'Fibrosis',
        'mass': 'Mass',
        'nodule': 'Nodule',
        'infiltration': 'Infiltration',
    }
    
    data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PadChest"):
        img_name = row['ImageID']
        
        # PadChest images are in folders 0-54
        # Format: 00000001_000.png -> folder 0
        folder_num = int(img_name.split('_')[0]) // 1000
        img_path = os.path.join(padchest_dir, str(folder_num), img_name)
        
        if not os.path.exists(img_path):
            continue
        
        # Labels are in 'Labels' column
        labels_text = str(row.get('Labels', '')).lower()
        label_vec = harmonize_labels_from_text(labels_text)
        
        data.append({
            'Image Index': img_name,
            'Processed Image Path': img_path,
            'Harmonized Labels': ' '.join(map(str, label_vec)),
            'Dataset': 'PadChest',
            'Split': 'train'
        })
    
    logger.info(f"Loaded {len(data)} PadChest samples")
    return data


def process_vindrcxr(dataset_dir):
    """Process VinDr-CXR (Vietnam) dataset"""
    logger.info("Processing VinDr-CXR...")
    
    vindr_dir = os.path.join(dataset_dir, 'VinDr-CXR')
    train_csv = os.path.join(vindr_dir, 'image_labels_train.csv')
    test_csv = os.path.join(vindr_dir, 'image_labels_test.csv')
    
    if not os.path.exists(train_csv):
        logger.warning("VinDr-CXR not found. Skipping.")
        logger.info("Download from: https://physionet.org/content/vindr-cxr/1.0.0/")
        return []
    
    data = []
    
    for csv_file, split_name in [(train_csv, 'train'), (test_csv, 'test')]:
        if not os.path.exists(csv_file):
            continue
        
        df = pd.read_csv(csv_file)
        
        # VinDr has detailed bounding box annotations
        # Group by image_id to get all findings per image
        for img_id, group in tqdm(df.groupby('image_id'), desc=f"VinDr-{split_name}"):
            img_path = os.path.join(vindr_dir, split_name, f"{img_id}.png")
            
            if not os.path.exists(img_path):
                # Try .jpg
                img_path = os.path.join(vindr_dir, split_name, f"{img_id}.jpg")
                if not os.path.exists(img_path):
                    continue
            
            label_vec = np.zeros(len(TARGET_PATHOLOGIES), dtype=int)
            
            # VinDr has 'class_name' column with findings
            for _, row in group.iterrows():
                finding = str(row.get('class_name', '')).lower()
                mapped_label = LABEL_MAPPING.get(finding)
                
                if mapped_label and mapped_label in TARGET_PATHOLOGIES:
                    idx = TARGET_PATHOLOGIES.index(mapped_label)
                    label_vec[idx] = 1
            
            data.append({
                'Image Index': f"{img_id}.png",
                'Processed Image Path': img_path,
                'Harmonized Labels': ' '.join(map(str, label_vec)),
                'Dataset': 'VinDr-CXR',
                'Split': split_name
            })
    
    logger.info(f"Loaded {len(data)} VinDr-CXR samples")
    return data


def process_rexgradient(dataset_dir):
    """Process ReXGradient dataset (existing implementation)"""
    logger.info("Processing ReXGradient...")
    
    rex_dir = os.path.join(dataset_dir, 'ReXGradient')
    if not os.path.exists(rex_dir):
        logger.warning("ReXGradient not found. Skipping.")
        return []
    
    # Use existing implementation from create_full_dataset.py
    # (Import and call the load_rex_data function)
    from create_full_dataset import load_rex_data
    return load_rex_data(rex_dir)


# ==================== MAIN PIPELINE ====================

def generate_complete_dataset(dataset_names=['all']):
    """Generate combined dataset from selected datasets"""
    
    # Detect dataset directory
    possible_paths = [
        "/home/mpi/Sharon/IIT/Dataset",  # HPC
        "D:/IIT/Dataset",                # Windows
        "../Dataset",                    # Relative
    ]
    
    DATASET_DIR = None
    for p in possible_paths:
        if os.path.exists(p):
            DATASET_DIR = os.path.abspath(p)
            break
    
    if not DATASET_DIR:
        logger.error("Could not find Dataset directory!")
        return
    
    logger.info(f"Using dataset directory: {DATASET_DIR}")
    
    # Process datasets
    processors = {
        'nih': lambda: process_nih(DATASET_DIR),
        'openi': lambda: process_openi(DATASET_DIR),
        'chexpert': lambda: process_chexpert(DATASET_DIR),
        'mimic': lambda: process_mimic(DATASET_DIR),
        'padchest': lambda: process_padchest(DATASET_DIR),
        'vindrcxr': lambda: process_vindrcxr(DATASET_DIR),
        'rexgradient': lambda: process_rexgradient(DATASET_DIR),
    }
    
    all_data = []
    stats = {}
    
    if 'all' in dataset_names:
        dataset_names = list(processors.keys())
    
    for name in dataset_names:
        if name in processors:
            data = processors[name]()
            all_data.extend(data)
            stats[name] = len(data)
        else:
            logger.warning(f"Unknown dataset: {name}")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "train_metadata_v2.csv")
    df.to_csv(output_path, index=False)
    
    # Statistics
    logger.info("=" * 70)
    logger.info("DATASET PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total samples: {len(df):,}")
    for name, count in stats.items():
        logger.info(f"  {name.upper()}: {count:,}")
    logger.info(f"\nSaved to: {output_path}")
    logger.info("=" * 70)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process multiple chest X-ray datasets')
    parser.add_argument('--datasets', nargs='+', default=['all'],
                       help='Datasets to process (all, nih, openi, chexpert, mimic, padchest, vindrcxr, rexgradient)')
    parser.add_argument('--list', action='store_true',
                       help='List available datasets')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable datasets:")
        print("  - nih: NIH ChestX-ray14 (~112K images)")
        print("  - openi: OpenI/Indiana (~7.5K images)")
        print("  - chexpert: CheXpert/Stanford (~224K images)")
        print("  - mimic: MIMIC-CXR/MIT (~377K images)")
        print("  - padchest: PadChest/Spain (~161K images)")
        print("  - vindrcxr: VinDr-CXR/Vietnam (~18K images)")
        print("  - rexgradient: ReXGradient (~160K images)")
        print("\nUsage: python create_full_dataset_v2.py --datasets all")
    else:
        generate_complete_dataset(args.datasets)

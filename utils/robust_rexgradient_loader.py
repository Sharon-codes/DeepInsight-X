# utils/robust_rexgradient_loader.py
import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from PIL import Image
import torch
from tqdm import tqdm
import json

# Import target pathologies from model_utils
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_utils import TARGET_PATHOLOGIES

def load_rexgradient_dataset_robust(save_dir="data/raw/ReXGradient-160K", max_samples_per_split=1000):
    """
    Load ReXGradient-160K dataset from HuggingFace with robust error handling.
    
    Args:
        save_dir (str): Directory to save the dataset locally
        max_samples_per_split (int): Maximum samples to process per split to avoid memory issues
        
    Returns:
        pd.DataFrame: Metadata dataframe with image paths and labels
    """
    print("Loading ReXGradient-160K dataset from HuggingFace (robust version)...")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    try:
        # Load dataset from HuggingFace
        ds = load_dataset("rajpurkarlab/ReXGradient-160K")
        
        print(f"Dataset loaded successfully. Available splits: {list(ds.keys())}")
        
        # Process each split
        all_metadata = []
        total_processed = 0
        total_errors = 0
        
        for split_name, split_data in ds.items():
            print(f"Processing {split_name} split with {len(split_data)} samples...")
            
            split_metadata = []
            processed_count = 0
            error_count = 0
            
            # Limit samples per split to avoid memory issues
            max_samples = min(len(split_data), max_samples_per_split)
            
            for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                if processed_count >= max_samples:
                    break
                    
                try:
                    # Check if image exists and is valid
                    if 'image' not in sample or sample['image'] is None:
                        error_count += 1
                        continue
                        
                    # Extract image and labels
                    image = sample['image']  # PIL Image
                    labels = sample['labels']  # List of pathology labels
                    report = sample.get('report', '')  # Radiology report (if available)
                    
                    # Validate image
                    if not isinstance(image, Image.Image):
                        error_count += 1
                        continue
                    
                    # Validate labels
                    if not isinstance(labels, list) or len(labels) == 0:
                        error_count += 1
                        continue
                    
                    # Generate filename
                    filename = f"rexgradient_{split_name}_{idx:06d}.png"
                    image_path = os.path.join(images_dir, filename)
                    
                    # Save image
                    image.save(image_path)
                    
                    # Create metadata entry
                    metadata_entry = {
                        'Image Index': filename,
                        'Dataset': 'ReXGradient-160K',
                        'Split': split_name,
                        'Processed Image Path': image_path,
                        'Original Labels': labels,
                        'Report': report,
                        'Harmonized Labels': harmonize_labels_to_string(labels)
                    }
                    
                    split_metadata.append(metadata_entry)
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if error_count % 100 == 0:  # Print error every 100 errors
                        print(f"Error processing sample {idx}: {str(e)}")
                    continue
            
            print(f"Split {split_name}: Processed {processed_count} samples, {error_count} errors")
            all_metadata.extend(split_metadata)
            total_processed += processed_count
            total_errors += error_count
        
        print(f"Total processed: {total_processed}, Total errors: {total_errors}")
        
        if total_processed == 0:
            print("No samples were successfully processed!")
            return None
        
        # Create metadata dataframe
        metadata_df = pd.DataFrame(all_metadata)
        
        # Save metadata
        metadata_path = os.path.join(save_dir, "metadata.csv")
        metadata_df.to_csv(metadata_path, index=False)
        print(f"Metadata saved to: {metadata_path}")
        
        return metadata_df
        
    except Exception as e:
        print(f"Error loading ReXGradient dataset: {str(e)}")
        return None

def harmonize_labels_to_string(labels):
    """
    Convert list of labels to harmonized string format.
    
    Args:
        labels (list): List of pathology labels
        
    Returns:
        str: Space-separated string of harmonized labels
    """
    # Create a mapping from common label variations to target pathologies
    label_mapping = {
        'atelectasis': 'Atelectasis',
        'cardiomegaly': 'Cardiomegaly', 
        'consolidation': 'Consolidation',
        'edema': 'Edema',
        'effusion': 'Effusion',
        'emphysema': 'Emphysema',
        'fibrosis': 'Fibrosis',
        'hernia': 'Hernia',
        'infiltration': 'Infiltration',
        'mass': 'Mass',
        'nodule': 'Nodule',
        'pleural_thickening': 'Pleural_Thickening',
        'pneumonia': 'Pneumonia',
        'pneumothorax': 'Pneumothorax'
    }
    
    # Initialize harmonized labels array
    harmonized = [0] * len(TARGET_PATHOLOGIES)
    
    # Map labels to target pathologies
    for label in labels:
        label_lower = label.lower().strip()
        if label_lower in label_mapping:
            target_label = label_mapping[label_lower]
            if target_label in TARGET_PATHOLOGIES:
                idx = TARGET_PATHOLOGIES.index(target_label)
                harmonized[idx] = 1
    
    return ' '.join(map(str, harmonized))

def process_rexgradient_dataset_robust(save_dir="data/raw/ReXGradient-160K", max_samples_per_split=1000):
    """
    Process ReXGradient dataset with robust error handling.
    
    Args:
        save_dir (str): Directory to save the dataset
        max_samples_per_split (int): Maximum samples per split
        
    Returns:
        pd.DataFrame: Processed metadata dataframe
    """
    print("Processing ReXGradient-160K dataset (robust version)...")
    
    # Check if already processed
    processed_path = "data/processed/rexgradient_processed_metadata.csv"
    if os.path.exists(processed_path):
        print("Loading existing processed ReXGradient data...")
        return pd.read_csv(processed_path)
    
    # Load and process dataset
    metadata_df = load_rexgradient_dataset_robust(save_dir, max_samples_per_split)
    
    if metadata_df is None:
        print("Failed to load ReXGradient dataset")
        return None
    
    # Create processed directory
    os.makedirs("data/processed", exist_ok=True)
    
    # Save processed metadata
    metadata_df.to_csv(processed_path, index=False)
    print(f"Processed metadata saved to: {processed_path}")
    
    return metadata_df

if __name__ == "__main__":
    # Test the robust loader
    df = process_rexgradient_dataset_robust(max_samples_per_split=100)
    if df is not None:
        print(f"Successfully processed {len(df)} samples")
        print("Sample data:")
        print(df.head())
    else:
        print("Failed to process dataset")

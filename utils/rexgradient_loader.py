# utils/rexgradient_loader.py
import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from PIL import Image
import torch
from tqdm import tqdm
import json

# Import target pathologies from model_utils
from utils.model_utils import TARGET_PATHOLOGIES

def load_rexgradient_dataset(save_dir="data/raw/ReXGradient-160K"):
    """
    Load ReXGradient-160K dataset from HuggingFace and save locally.
    
    Args:
        save_dir (str): Directory to save the dataset locally
        
    Returns:
        pd.DataFrame: Metadata dataframe with image paths and labels
    """
    print("Loading ReXGradient-160K dataset from HuggingFace...")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    try:
        # Load dataset from HuggingFace
        # Note: You may need to login using `huggingface-cli login` to access this dataset
        ds = load_dataset("rajpurkarlab/ReXGradient-160K")
        
        print(f"Dataset loaded successfully. Available splits: {list(ds.keys())}")
        
        # Process each split
        all_metadata = []
        
        for split_name, split_data in ds.items():
            print(f"Processing {split_name} split with {len(split_data)} samples...")
            
            split_metadata = []
            
            for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                try:
                    # Extract image and labels
                    image = sample['image']  # PIL Image
                    labels = sample['labels']  # List of pathology labels
                    report = sample.get('report', '')  # Radiology report (if available)
                    
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
                        'Original Labels': labels,
                        'Report': report,
                        'Image Path': image_path
                    }
                    
                    split_metadata.append(metadata_entry)
                    
                except Exception as e:
                    print(f"Error processing sample {idx} in {split_name}: {e}")
                    continue
            
            # Convert to DataFrame
            split_df = pd.DataFrame(split_metadata)
            split_df.to_csv(os.path.join(save_dir, f"{split_name}_metadata.csv"), index=False)
            all_metadata.extend(split_metadata)
            
            print(f"Saved {len(split_metadata)} samples from {split_name} split")
        
        # Create combined metadata
        combined_df = pd.DataFrame(all_metadata)
        combined_df.to_csv(os.path.join(save_dir, "combined_metadata.csv"), index=False)
        
        print(f"ReXGradient-160K dataset loaded successfully!")
        print(f"Total samples: {len(combined_df)}")
        print(f"Images saved to: {images_dir}")
        print(f"Metadata saved to: {save_dir}")
        
        return combined_df
        
    except Exception as e:
        print(f"Error loading ReXGradient-160K dataset: {e}")
        print("Make sure you have logged in to HuggingFace using: huggingface-cli login")
        return None

def harmonize_rexgradient_labels(metadata_df):
    """
    Harmonize ReXGradient labels to the standard TARGET_PATHOLOGIES.
    
    Args:
        metadata_df (pd.DataFrame): ReXGradient metadata dataframe
        
    Returns:
        pd.DataFrame: Updated dataframe with harmonized labels
    """
    print("Harmonizing ReXGradient labels...")
    
    num_samples = len(metadata_df)
    num_target_labels = len(TARGET_PATHOLOGIES)
    harmonized_labels = np.zeros((num_samples, num_target_labels), dtype=int)
    
    # Create mapping from ReXGradient labels to our target pathologies
    # This mapping may need to be adjusted based on the actual ReXGradient label names
    label_mapping = {
        # Map common variations to our standard labels
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
        'pneumothorax': 'Pneumothorax',
        # Add more mappings as needed
        'pleural effusion': 'Effusion',
        'lung opacity': 'Consolidation',
        'lung nodule': 'Nodule',
        'chest mass': 'Mass',
        'cardiac enlargement': 'Cardiomegaly',
        'pulmonary edema': 'Edema',
        'atelectatic changes': 'Atelectasis',
        'fibrotic changes': 'Fibrosis',
        'emphysematous changes': 'Emphysema',
        'pleural thickening': 'Pleural_Thickening',
        'pneumonic consolidation': 'Pneumonia',
        'pneumothorax': 'Pneumothorax'
    }
    
    label_to_idx = {label: i for i, label in enumerate(TARGET_PATHOLOGIES)}
    
    for idx, row in metadata_df.iterrows():
        original_labels = row['Original Labels']
        
        if isinstance(original_labels, list):
            # Process list of labels
            for label in original_labels:
                if isinstance(label, str):
                    label_lower = label.lower().strip()
                    
                    # Try direct mapping first
                    if label_lower in label_mapping:
                        target_label = label_mapping[label_lower]
                        if target_label in label_to_idx:
                            harmonized_labels[idx, label_to_idx[target_label]] = 1
                    else:
                        # Try partial matching for complex labels
                        for rex_label, target_label in label_mapping.items():
                            if rex_label in label_lower or label_lower in rex_label:
                                if target_label in label_to_idx:
                                    harmonized_labels[idx, label_to_idx[target_label]] = 1
                                    break
        
        elif isinstance(original_labels, str):
            # Process string labels (comma-separated or pipe-separated)
            labels = original_labels.replace('|', ',').split(',')
            for label in labels:
                label_lower = label.lower().strip()
                
                if label_lower in label_mapping:
                    target_label = label_mapping[label_lower]
                    if target_label in label_to_idx:
                        harmonized_labels[idx, label_to_idx[target_label]] = 1
    
    # Convert harmonized labels to string format for consistency with other datasets
    harmonized_labels_str = []
    for i in range(num_samples):
        label_str = ' '.join(map(str, harmonized_labels[i]))
        harmonized_labels_str.append(label_str)
    
    metadata_df['Harmonized Labels'] = harmonized_labels_str
    
    # Print label distribution
    print("Label distribution in ReXGradient dataset:")
    for i, pathology in enumerate(TARGET_PATHOLOGIES):
        count = np.sum(harmonized_labels[:, i])
        percentage = (count / num_samples) * 100
        print(f"  {pathology}: {count} samples ({percentage:.2f}%)")
    
    return metadata_df

def process_rexgradient_dataset(save_dir="data/raw/ReXGradient-160K", processed_dir="data/processed"):
    """
    Complete processing pipeline for ReXGradient-160K dataset.
    
    Args:
        save_dir (str): Directory to save raw ReXGradient data
        processed_dir (str): Directory to save processed data
        
    Returns:
        pd.DataFrame: Processed metadata dataframe
    """
    # Load dataset
    metadata_df = load_rexgradient_dataset(save_dir)
    
    if metadata_df is None:
        print("Failed to load ReXGradient dataset")
        return None
    
    # Harmonize labels
    metadata_df = harmonize_rexgradient_labels(metadata_df)
    
    # Process images (resize to standard size)
    from utils.preprocessing import process_and_save_images
    
    processed_images_dir = os.path.join(processed_dir, "images")
    os.makedirs(processed_images_dir, exist_ok=True)
    
    print("Processing ReXGradient images...")
    processed_df = process_and_save_images(
        os.path.join(save_dir, "images"),
        processed_images_dir,
        metadata_df,
        dataset_type_for_dicom_processing=None  # Images are already PNGs
    )
    
    # Save processed metadata
    processed_df.to_csv(os.path.join(processed_dir, "rexgradient_processed_metadata.csv"), index=False)
    
    print(f"ReXGradient processing complete!")
    print(f"Processed {len(processed_df)} samples")
    
    return processed_df

if __name__ == "__main__":
    # Example usage
    print("Processing ReXGradient-160K dataset...")
    processed_df = process_rexgradient_dataset()
    
    if processed_df is not None:
        print("ReXGradient-160K dataset processing completed successfully!")
        print(f"Final dataset shape: {processed_df.shape}")
        print(f"Columns: {list(processed_df.columns)}")
    else:
        print("ReXGradient-160K dataset processing failed!")

#!/usr/bin/env python3
"""
Script to find and remove corrupted PNG images from the processed dataset.
Run this before training to ensure all images are valid.
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm

def check_and_clean_images():
    """Check all processed images and remove corrupted ones."""
    
    metadata_file = "data/processed/nih_full_metadata.csv"
    
    if not os.path.exists(metadata_file):
        print(f"Metadata file not found: {metadata_file}")
        return
    
    print("Loading metadata...")
    df = pd.read_csv(metadata_file)
    
    corrupted_images = []
    valid_indices = []
    
    print(f"Checking {len(df)} images for corruption...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
        img_path = row['Processed Image Path']
        
        # Try to load the image
        try:
            image = cv2.imread(img_path)
            
            if image is None or image.size == 0:
                corrupted_images.append(img_path)
                print(f"\nCorrupted: {img_path}")
            else:
                valid_indices.append(idx)
        except Exception as e:
            corrupted_images.append(img_path)
            print(f"\nError loading {img_path}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Validation Complete:")
    print(f"  Total images: {len(df)}")
    print(f"  Valid images: {len(valid_indices)}")
    print(f"  Corrupted images: {len(corrupted_images)}")
    print(f"{'='*60}")
    
    if corrupted_images:
        print("\nCorrupted images will be removed from the dataset.")
        
        # Remove corrupted images from metadata
        df_clean = df.iloc[valid_indices].reset_index(drop=True)
        
        # Backup original metadata
        backup_file = metadata_file.replace('.csv', '_backup.csv')
        df.to_csv(backup_file, index=False)
        print(f"✓ Original metadata backed up to: {backup_file}")
        
        # Save cleaned metadata
        df_clean.to_csv(metadata_file, index=False)
        print(f"✓ Cleaned metadata saved to: {metadata_file}")
        
        # Delete corrupted image files
        for img_path in corrupted_images:
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"  Deleted: {img_path}")
        
        print(f"\n✓ Removed {len(corrupted_images)} corrupted images")
    else:
        print("\n✓ All images are valid!")

if __name__ == "__main__":
    check_and_clean_images()

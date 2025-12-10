import torch
from utils.data_loader import ChestXrayDataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def debug_dataset():
    print("Debugging ChestXrayDataset...")
    
    # Paths
    processed_dir = "data/processed"
    train_csv = os.path.join(processed_dir, "train_metadata.csv")
    image_dir = os.path.join(processed_dir, "images")
    
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found!")
        return

    # 1. Check CSV content directly
    print(f"\n1. Checking CSV content ({train_csv})...")
    df = pd.read_csv(train_csv)
    print(f"Total samples: {len(df)}")
    print("First 5 rows 'Harmonized Labels':")
    print(df['Harmonized Labels'].head().tolist())
    
    # Check if labels look valid (not all same)
    unique_labels = df['Harmonized Labels'].unique()
    print(f"Number of unique label combinations: {len(unique_labels)}")
    if len(unique_labels) < 2:
        print("WARNING: All labels seem to be identical!")
        
    # 2. Check Dataset class loading
    print(f"\n2. Checking ChestXrayDataset loading...")
    try:
        dataset = ChestXrayDataset(train_csv, image_dir, is_train=False) # is_train=False to skip augmentations for inspection
        print(f"Dataset length: {len(dataset)}")
        
        # Check first 5 samples
        print("\nChecking first 5 samples from Dataset:")
        for i in range(5):
            img, label = dataset[i]
            print(f"Sample {i}: Label shape: {label.shape}, Label values: {label}")
            print(f"          Image shape: {img.shape}, Max val: {img.max()}, Min val: {img.min()}")
            
            if torch.all(label == 0):
                print("  -> WARNING: Label is all zeros (No findings)")
                
        # Check class distribution in Dataset
        print("\nChecking class distribution in first 1000 samples...")
        all_labels = []
        for i in tqdm(range(min(1000, len(dataset)))):
            _, label = dataset[i]
            all_labels.append(label.numpy())
            
        all_labels = np.array(all_labels)
        class_counts = np.sum(all_labels, axis=0)
        print(f"Class counts (first {len(all_labels)} samples): {class_counts}")
        
        if np.sum(class_counts) == 0:
            print("CRITICAL ERROR: No pathologies found in the checked samples! Labels are likely broken.")
        else:
            print("Labels seem to be loading with some positive values.")

    except Exception as e:
        print(f"Error initializing or iterating dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dataset()

# utils/data_loader.py
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2 # For Albumentations

# Import preprocessing functions
from utils.preprocessing import create_transforms, TARGET_IMAGE_SIZE

class ChestXrayDataset(Dataset):
    def __init__(self, metadata_path, image_dir, is_train=True):
        """
        Args:
            metadata_path (str): Path to the CSV file containing image paths and labels.
            image_dir (str): Directory containing the processed images.
            is_train (bool): Whether this is for training (applies augmentation) or validation/test.
        """
        self.metadata_df = pd.read_csv(metadata_path)
        self.image_dir = image_dir
        self.transform = create_transforms(is_train=is_train, image_size=TARGET_IMAGE_SIZE)
        
        # Convert 'Harmonized Labels' column from string representation of list/array to actual numpy array
        # This is crucial if the CSV saves lists as strings like "[0 1 0 0]"
        self.labels = []
        for _, row in self.metadata_df.iterrows():
            label_str = row['Harmonized Labels']
            # Example: "[0 1 0 0]" or ""
            if isinstance(label_str, str):
                # Remove brackets and split by space or comma, then convert to int
                # Handle empty strings or "" for cases with no findings
                clean_label_str = label_str.strip(' ').replace(',', ' ')
                if clean_label_str:
                    label_array = np.array([int(x) for x in clean_label_str.split() if x.strip()])
                else: # If no labels, create an array of zeros with the correct number of classes
                    # This assumes all label arrays have the same length (num_classes)
                    # You might need to infer num_classes from the first non-empty label or a global constant
                    label_array = np.zeros(14, dtype=int) # 14 classes for chest X-ray pathologies
            else: # Assume it's already a list or numpy array
                label_array = np.array(label_str)
            self.labels.append(label_array)
        self.labels = np.array(self.labels) # Convert list of arrays to a single 2D numpy array

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        # Handle different column names for image path
        if 'Processed Image Path' in self.metadata_df.columns:
            img_path = self.metadata_df.iloc[idx]['Processed Image Path']
        elif 'Image Path' in self.metadata_df.columns:
            img_path = self.metadata_df.iloc[idx]['Image Path']
        else:
            # Fallback to 'Image Index' if neither path column exists
            img_path = self.metadata_df.iloc[idx]['Image Index']
            
        full_img_path = os.path.join(self.image_dir, os.path.basename(img_path)) # Use basename as image_dir is already processed
        
        # Load image using OpenCV for Albumentations compatibility
        try:
            image = cv2.imread(full_img_path)
            
            # Check if image loaded successfully
            if image is None or image.size == 0:
                raise ValueError(f"Failed to load image: {full_img_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
            
        except Exception as e:
            # If image is corrupted or missing, create a black placeholder
            print(f"Warning: Error loading image {full_img_path}: {e}. Using black placeholder.")
            # Create a black image of the expected size
            from utils.preprocessing import TARGET_IMAGE_SIZE
            image = np.zeros((TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3), dtype=np.uint8)
        
        # Get multi-hot encoded labels
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image'] # Albumentations returns a dictionary

        return image, label

def get_data_loaders(processed_data_dir, train_metadata_path, val_metadata_path, test_metadata_path, batch_size, num_workers, world_size=1, rank=0):
    """
    Creates PyTorch DataLoaders for training, validation, and testing.
    Uses DistributedSampler for DDP training.
    [19, 20, 21]
    """
    # Assuming all processed images are in `processed_data_dir/images`
    image_base_dir = os.path.join(processed_data_dir, 'images')

    train_dataset = ChestXrayDataset(train_metadata_path, image_base_dir, is_train=True)
    val_dataset = ChestXrayDataset(val_metadata_path, image_base_dir, is_train=False)
    test_dataset = ChestXrayDataset(test_metadata_path, image_base_dir, is_train=False)

    # DistributedSampler for DDP training [19]
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None), # Only shuffle if not using DistributedSampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True # Speeds up data transfer to GPU [19]
    )

    # Validation and test loaders typically don't use DistributedSampler unless
    # you're doing distributed evaluation, which is more complex.
    # For simplicity, validation/test is often run on rank 0 or a single GPU.
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset.labels.shape[1] # Return num_classes
# utils/preprocessing.py
import os
import pydicom
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2 # For converting NumPy array to PyTorch tensor

# Define target image resolution for CNN input
TARGET_IMAGE_SIZE = (1024, 1024) # Fixed resolution for all images [2, 3, 4]

# ImageNet normalization parameters (common for pre-trained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def dicom_to_png(dicom_path, output_path, apply_openi_specific_preprocessing=False):
    """
    Converts a DICOM file to a PNG image, handling pixel intensity normalization.
    Optionally applies specific preprocessing steps from OpenI dataset (clipping, linear scaling).
   
    """
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        pixel_array = dicom_data.pixel_array
        
        # Apply VOI LUT if available for proper windowing
        if 'WindowCenter' in dicom_data and 'WindowWidth' in dicom_data:
            pixel_array = pydicom.pixel_data_handlers.util.apply_voi_lut(pixel_array, dicom_data)
        
        # Handle MONOCHROME1 (inverted grayscale) to MONOCHROME2 if necessary [5]
        if 'PhotometricInterpretation' in dicom_data and dicom_data.PhotometricInterpretation == 'MONOCHROME1':
            pixel_array = np.amax(pixel_array) - pixel_array # Invert pixel values
        
        # Apply OpenI specific preprocessing: clipping and linear scaling
        if apply_openi_specific_preprocessing:
            # Clip top/bottom 0.5% DICOM pixel values
            lower_bound = np.percentile(pixel_array, 0.5)
            upper_bound = np.percentile(pixel_array, 99.5)
            pixel_array = np.clip(pixel_array, lower_bound, upper_bound)
            
            # Scale linearly to fit into 0-255 range after clipping
            # Ensure min/max are not identical to avoid division by zero if image is flat
            min_val = pixel_array.min()
            max_val = pixel_array.max()
            if max_val == min_val: # Handle flat images
                pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
            else:
                pixel_array = (pixel_array - min_val) / (max_val - min_val) * 255
                pixel_array = pixel_array.astype(np.uint8)
        else:
            # Default normalization to 0-255 if not OpenI specific
            if pixel_array.dtype!= np.uint8:
                pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        image = Image.fromarray(pixel_array)
        # The "resized to 2048 on shorter side" is an *intermediate* step from the original OpenI processing.
        # Our pipeline will handle the final TARGET_IMAGE_SIZE (1024x1024) in process_and_save_images.
        image.save(output_path)
        return True
    except Exception as e:
        print(f"Error converting DICOM {dicom_path} to PNG: {e}")
        return False

def create_transforms(is_train=True, image_size=TARGET_IMAGE_SIZE):
    """
    Defines advanced image augmentation pipelines using Albumentations.
   
    """
    if is_train:
        return A.Compose([
            A.Resize(image_size[0], image_size[1], interpolation=cv2.INTER_AREA), # Resize first [6, 7]
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(p=0.2), # Simulate sensor noise
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=0.2), # Simulate camera focus variations
                A.MotionBlur(blur_limit=3, p=0.3),
            ], p=0.2),
            A.CoarseDropout(max_holes=8, max_height=image_size[0]//20, max_width=image_size[1]//20,
                            min_holes=1, fill_value=0, mask_fill_value=0, p=0.2), # Regularization
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), # Normalize pixel values [8, 2, 9]
            ToTensorV2() # Convert to PyTorch tensor [10]
        ])
    else: # Validation/Test transforms (no heavy augmentation)
        return A.Compose([
            A.Resize(image_size[0], image_size[1], interpolation=cv2.INTER_AREA), # Resize first [6, 7]
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])

def process_and_save_images(input_dir, output_dir, metadata_df, dataset_type_for_dicom_processing=None):
    """
    Processes images (DICOM/PNG) and saves them to a target directory with fixed resolution.
    Also updates metadata with new image paths.
    dataset_type_for_dicom_processing: String indicating specific DICOM preprocessing to apply (e.g., 'OpenI').
    [8, 6, 11, 12]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # This transform will handle the final resize to TARGET_IMAGE_SIZE for all images
    process_transform = A.Compose([
        A.Resize(TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], interpolation=cv2.INTER_AREA), # Resize to fixed CNN input size [6, 7]
        ToTensorV2() # Convert to tensor temporarily for consistent handling
    ])

    processed_image_paths = []
    original_image_indices = []

    print(f"Processing images from {input_dir} to {output_dir}...")
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        original_image_filename = row['Image Index'] # Assuming 'Image Index' column exists
        original_image_path = os.path.join(input_dir, original_image_filename)
        
        # Determine output filename (e.g., keep original name but ensure.png extension)
        base_filename, ext = os.path.splitext(original_image_filename)
        output_filename = f"{base_filename}.png"
        output_filepath = os.path.join(output_dir, output_filename)

        if not os.path.exists(original_image_path):
            print(f"Warning: Image not found at {original_image_path}. Skipping.")
            continue

        success = False
        if ext.lower() == '.dcm':
            # Apply OpenI specific DICOM preprocessing if specified
            success = dicom_to_png(original_image_path, output_filepath, 
                                   apply_openi_specific_preprocessing=(dataset_type_for_dicom_processing == 'OpenI'))
        elif ext.lower() in ['.png', '.jpg', '.jpeg']:
            try:
                # Load, resize, and save as PNG
                img = Image.open(original_image_path).convert('RGB')
                img_np = np.array(img)
                transformed_img = process_transform(image=img_np)['image']
                # Convert back to PIL Image for saving as PNG
                Image.fromarray(transformed_img.permute(1, 2, 0).numpy().astype(np.uint8)).save(output_filepath)
                success = True
            except Exception as e:
                print(f"Error processing image {original_image_path}: {e}")
                success = False
        else:
            print(f"Unsupported file format for {original_image_filename}. Skipping.")
            success = False
        
        if success:
            processed_image_paths.append(output_filepath)
            original_image_indices.append(idx)
    
    # Create a new DataFrame with processed image paths and original labels
    processed_metadata_df = metadata_df.loc[original_image_indices].copy()
    processed_metadata_df['Processed Image Path'] = processed_image_paths
    
    return processed_metadata_df

def harmonize_labels(metadata_df, target_labels):
    """
    Harmonizes diverse labels to a unified set of target labels.
    This is a conceptual implementation; actual mapping rules depend on datasets.
    [13, 14, 15, 16, 17, 18]
    """
    print("Harmonizing labels...")
    
    num_samples = len(metadata_df)
    num_target_labels = len(target_labels)
    harmonized_labels = np.zeros((num_samples, num_target_labels), dtype=int)
    
    label_to_idx = {label: i for i, label in enumerate(target_labels)}

    for idx, row in tqdm(metadata_df.iterrows(), total=num_samples):
        dataset_name = row.get('Dataset', 'Unknown') # Get dataset name for specific mapping
        
        # This is a placeholder for complex, dataset-specific label mapping logic.
        # You would need to define rules for each dataset (NIH, MIMIC, CheXpert, PadChest, OpenI, etc.)
        # based on their original label sets and how they map to your TARGET_PATHOLOGIES.
        
        if dataset_name == 'NIH ChestX-ray14':
            if 'Finding Labels' in row and isinstance(row['Finding Labels'], str):
                current_labels = row['Finding Labels'].split('|')
                for label in current_labels:
                    if label in label_to_idx:
                        harmonized_labels[idx, label_to_idx[label]] = 1
                    elif label == 'No Finding' and 'No Finding' in label_to_idx:
                        # Handle 'No Finding' if it's a specific target label
                        harmonized_labels[idx, label_to_idx['No Finding']] = 1
        elif dataset_name == 'OpenI':
            # For OpenI, you'd typically parse its XML reports or a derived metadata CSV for findings.
            # This is a highly simplified example. Real implementation requires robust NLP for OpenI reports.
            # Assuming 'Report Impression' or similar column exists in your OpenI metadata.csv
            if 'Report Impression' in row and isinstance(row['Report Impression'], str):
                report_text = row.lower()
                if 'pneumonia' in report_text and 'Pneumonia' in label_to_idx:
                    harmonized_labels[idx, label_to_idx['Pneumonia']] = 1
                if 'cardiomegaly' in report_text and 'Cardiomegaly' in label_to_idx:
                    harmonized_labels[idx, label_to_idx['Cardiomegaly']] = 1
                # Add more rules for other pathologies based on keywords or more advanced NLP
            # If OpenI has structured labels in its metadata, use those directly.
        elif dataset_name == 'ReXGradient-160K':
            # For ReXGradient-160K, labels are already harmonized in the loader
            # The 'Harmonized Labels' column should contain the processed labels
            if 'Harmonized Labels' in row and isinstance(row['Harmonized Labels'], str):
                label_str = row['Harmonized Labels']
                if label_str.strip():
                    label_array = np.array([int(x) for x in label_str.split() if x.strip()])
                    # Ensure the array has the correct length
                    if len(label_array) == num_target_labels:
                        harmonized_labels[idx] = label_array
                    else:
                        print(f"Warning: Label array length mismatch for sample {idx}")
            # If no harmonized labels, keep as zeros (no findings)

    metadata_df['Harmonized Labels'] = list(harmonized_labels)
    
    return metadata_df

if __name__ == '__main__':
    # Example usage for preprocessing a small subset of NIH ChestX-ray14 and OpenI
    # In a real scenario, you would download and extract all datasets first.
    
    # Dummy setup for demonstration
    print("Creating dummy data for demonstration...")
    os.makedirs('data/raw/NIH_ChestXray14/images', exist_ok=True)
    os.makedirs('data/raw/OpenI/images', exist_ok=True)

    # Dummy NIH data
    nih_dummy_image_paths = []
    nih_dummy_labels = []
    for i in range(50):
        dummy_img = Image.new('RGB', (512, 512), color = (i % 255, (i+50) % 255, (i+100) % 255))
        img_filename = f"nih_image_{i:03d}.png"
        dummy_img.save(os.path.join('data/raw/NIH_ChestXray14/images', img_filename))
        nih_dummy_image_paths.append(img_filename)
        labels = []
        if i % 3 == 0: labels.append('Atelectasis')
        if i % 5 == 0: labels.append('Pneumonia')
        if not labels: labels.append('No Finding')
        nih_dummy_labels.append('|'.join(labels))
    nih_dummy_metadata = pd.DataFrame({
        'Image Index': nih_dummy_image_paths,
        'Finding Labels': nih_dummy_labels
    })
    nih_dummy_metadata.to_csv('data/raw/NIH_ChestXray14/Data_entry_2017.csv', index=False)

    # Dummy OpenI data (simulating already processed PNGs from Kaggle)
    openi_dummy_image_paths = []
    openi_dummy_reports = [] # Simulating text reports for OpenI
    for i in range(50):
        # Simulating 2048 shorter side, will be resized to 1024x1024 later
        dummy_img = Image.new('RGB', (2048, 2500), color = ((i+100) % 255, (i+150) % 255, (i+200) % 255))
        img_filename = f"openi_image_{i:03d}.png"
        dummy_img.save(os.path.join('data/raw/OpenI/images', img_filename))
        openi_dummy_image_paths.append(img_filename)
        report_text = "Normal chest x-ray. No acute cardiopulmonary abnormality."
        if i % 4 == 0: report_text = "Findings suggestive of pneumonia in the right lower lobe."
        if i % 6 == 0: report_text = "Mild cardiomegaly noted."
        openi_dummy_reports.append(report_text)
    openi_dummy_metadata = pd.DataFrame({
        'Image Index': openi_dummy_image_paths,
        'Report Impression': openi_dummy_reports # Assuming this column for OpenI reports
    })
    openi_dummy_metadata.to_csv('data/raw/OpenI/metadata.csv', index=False)
    print("Dummy data created.")

    # --- Main preprocessing pipeline ---
    PROCESSED_DATA_DIR = 'data/processed' # This is the parent dir for images and metadata
    PROCESSED_IMAGES_SUBDIR = os.path.join(PROCESSED_DATA_DIR, 'images')
    os.makedirs(PROCESSED_IMAGES_SUBDIR, exist_ok=True)

    # Define your 14 target pathologies (must be consistent with model_utils.py)
    TARGET_PATHOLOGIES = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
        'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
        'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
    ]

    all_processed_metadata = []

    # --- NIH ChestX-ray14 Processing ---
    NIH_RAW_DATA_DIR = 'data/raw/NIH_ChestXray14/images' # Adjust this to your actual raw image directory
    NIH_METADATA_PATH = 'data/raw/NIH_ChestXray14/Data_entry_2017.csv'
    
    print("Processing NIH ChestX-ray14 images...")
    nih_original_metadata_df = pd.read_csv(NIH_METADATA_PATH)
    nih_original_metadata_df['Dataset'] = 'NIH ChestX-ray14' # Add dataset identifier
    nih_processed_df = process_and_save_images(NIH_RAW_DATA_DIR, PROCESSED_IMAGES_SUBDIR, nih_original_metadata_df, dataset_type_for_dicom_processing='NIH')
    all_processed_metadata.append(nih_processed_df)

    # --- OpenI Dataset Processing ---
    # For OpenI, the Kaggle version already has PNGs processed with clipping/scaling to 2048 shorter side.
    # So, if using the Kaggle version, you would treat them as regular PNGs that need final resize to 1024x1024.
    # If you downloaded raw DICOMs from NLM OpenI, you'd use `dataset_type_for_dicom_processing='OpenI'`.
    OPENI_RAW_DATA_DIR = 'data/raw/OpenI/images'
    OPENI_METADATA_PATH = 'data/raw/OpenI/metadata.csv'
    
    print("Processing OpenI images...")
    openi_original_metadata_df = pd.read_csv(OPENI_METADATA_PATH)
    openi_original_metadata_df['Dataset'] = 'OpenI' # Add dataset identifier
    # IMPORTANT: Set `dataset_type_for_dicom_processing='OpenI'` if you are processing raw DICOMs from OpenI.
    # If your OpenI images are already PNGs from Kaggle, set it to `None`.
    openi_processed_df = process_and_save_images(OPENI_RAW_DATA_DIR, PROCESSED_IMAGES_SUBDIR, openi_original_metadata_df, dataset_type_for_dicom_processing=None) # Set to 'OpenI' if raw DICOMs
    all_processed_metadata.append(openi_processed_df)

    # --- Add processing for other datasets (MIMIC-CXR, CheXpert, PadChest, ReXGradient-160K, CheXMask) here ---
    # You would add similar blocks for each dataset, adjusting paths and metadata loading.
    # Ensure each dataset's metadata is loaded and a 'Dataset' column is added for tracking.

    # Combine all processed metadata
    combined_processed_metadata_df = pd.concat(all_processed_metadata, ignore_index=True)

    # Harmonize labels for the combined dataset
    final_metadata_df = harmonize_labels(combined_processed_metadata_df, TARGET_PATHOLOGIES)

    # Save the processed metadata for later use by the DataLoader
    final_metadata_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'combined_metadata.csv'), index=False)
    # IMPORTANT: You must manually split this `combined_metadata.csv` into
    # `train_metadata.csv`, `val_metadata.csv`, and `test_metadata.csv`
    # for your training, validation, and test sets. Ensure stratified sampling.
    print(f"Combined and processed metadata saved to {os.path.join(PROCESSED_DATA_DIR, 'combined_metadata.csv')}")
    print("Preprocessing complete.")
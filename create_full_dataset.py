"""
Generate complete corrected dataset from scratch
"""
import pandas as pd
import numpy as np
# from datasets import load_dataset
from tqdm import tqdm
import logging
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Correct NIH index mapping (verified from previous run)
INDEX_TO_PATHOLOGY = {
    0: 'Atelectasis',
    1: 'Cardiomegaly', 
    2: 'Effusion',
    3: 'Infiltration',
    4: 'Mass',
    5: 'Nodule',
    6: 'Pneumonia',
    7: 'Pneumothorax',
    8: 'Consolidation',
    9: 'Edema',
    10: 'Emphysema',
    11: 'Fibrosis',
    12: 'Pleural_Thickening',
    13: 'Hernia',
    14: 'No Finding'
}

TARGET_PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

OUTPUT_DIR = "data/processed"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")


# --- OpenI Processing ---
def load_openi_data(reports_path, projections_path, images_dir):
    if not os.path.exists(reports_path) or not os.path.exists(projections_path):
        logger.warning("OpenI CSVs not found. Skipping OpenI.")
        return []

    logger.info("Loading OpenI dataset...")
    reports = pd.read_csv(reports_path)
    projections = pd.read_csv(projections_path)
    
    # Merge on uid
    merged = pd.merge(reports, projections, on='uid', how='inner')
    
    data = []
    
    # Mapping
    # Terms to NIH Labels (14 targets)
    # Rules: simple keyword matching for now
    mapping = {
        'atelectasis': 'Atelectasis',
        'cardiomegaly': 'Cardiomegaly',
        'effusion': 'Effusion',
        'infiltration': 'Infiltration',
        'mass': 'Mass',
        'nodule': 'Nodule',
        'pneumonia': 'Pneumonia',
        'pneumothorax': 'Pneumothorax',
        'consolidation': 'Consolidation',
        'edema': 'Edema',
        'emphysema': 'Emphysema',
        'fibrosis': 'Fibrosis',
        'thickening': 'Pleural_Thickening',
        'hernia': 'Hernia',
        # Synonyms/Related
        'granuloma': 'Nodule',
        'opacity': 'Infiltration',
        'degenerative': 'No Finding', # Ignore
        'normal': 'No Finding'
    }

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Processing OpenI"):
        img_filename = row['filename']
        img_path = os.path.join(images_dir, img_filename)
        
        if not os.path.exists(img_path):
            continue
            
        # Parse labels
        text_labels = str(row['MeSH']).split(';') + str(row['Problems']).split(';')
        text_labels = [t.lower().strip() for t in text_labels if t != 'nan']
        
        label_vec = np.zeros(len(TARGET_PATHOLOGIES), dtype=int)
        
        found_finding = False
        for term in text_labels:
            for key, target in mapping.items():
                if key in term:
                    if target != 'No Finding':
                        idx = TARGET_PATHOLOGIES.index(target)
                        label_vec[idx] = 1
                        found_finding = True
        
        # If no findings mapped but report implies normal, set No Finding (implicit by all-zeros)
        # But we verify if 'normal' was explicitly there or just no map
        if not found_finding and 'normal' in ' '.join(text_labels):
             pass # All zeros is "No Finding" in implicit multi-hot if we treat 14th class separately?
             # Wait, NIH has explicit 'No Finding' class often.
             # The current script uses 14 classes?
             # TARGET_PATHOLOGIES has 14 items. 'No Finding' is NOT in TARGET_PATHOLOGIES.
             # So all-zeros = No Finding. Correct.
        
        data.append({
            'Image Index': img_filename,
            'Processed Image Path': img_path,
            'Harmonized Labels': ' '.join(map(str, label_vec)),
            'Dataset': 'OpenI'
        })
        
    logger.info(f"Loaded {len(data)} OpenI samples.")
    return data


def get_nih_image_paths(dataset_dir):
    """Scan for all NIH images in local subdirectories."""
    image_paths = {}
    # Folders are images_001 to images_012, containing 'images' subfolder
    # Or just 'images' folder
    # Based on file listing, we have images_001/images, etc.
    # Also 'images' folder might exist.
    
    search_dirs = [f"images_{i:03d}" for i in range(1, 13)] + ['images']
    
    print("Scanning local NIH images...")
    for subdir in search_dirs:
        full_dir = os.path.join(dataset_dir, subdir, 'images')
        if os.path.exists(full_dir):
            for fname in os.listdir(full_dir):
                if fname.endswith('.png'):
                    image_paths[fname] = os.path.join(full_dir, fname)
                    
    print(f"Found {len(image_paths)} local NIH images.")
    return image_paths


# --- ReX Data Processing ---
def load_rex_data(rex_dir):
    """
    Load ReX dataset. Supports JSON metadata and multi-part image folders.
    """
    if not os.path.exists(rex_dir):
        logger.warning(f"ReX directory {rex_dir} not found. Skipping.")
        return []

    logger.info(f"Scanning ReX directory: {rex_dir}")
    
    # 1. Load Metadata
    # ReX typically has 'metadata/train_metadata.json' or 'train.json'
    metadata_path = None
    possible_metas = [
        os.path.join(rex_dir, "metadata", "train_metadata.json"),
        os.path.join(rex_dir, "train_metadata.json"),
        os.path.join(rex_dir, "metadata.csv") # Legacy
    ]
    
    for p in possible_metas:
        if os.path.exists(p):
            metadata_path = p
            break
            
    if not metadata_path:
        # Fallback to searching
        for root, _, files in os.walk(rex_dir):
            for f in files:
                if f.endswith('metadata.json') or f == 'train.csv':
                    metadata_path = os.path.join(root, f)
                    break
            if metadata_path: break
            
    if not metadata_path:
        logger.warning("No valid metadata file found for ReX. Skipping.")
        return []
        
    logger.info(f"Loading ReX metadata from: {metadata_path}")
    
    # Debug: peek at file content
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            head = f.read(500)
            print(f"File Header Preview:\n{head}\n...")
    except:
        pass

    # Load DataFrame
    df = pd.DataFrame()
    try:
        # Strategy 1: Dictionary-based JSON (orient='index')
        # The preview showed key-value pairs where key is ID.
        with open(metadata_path, 'r') as f:
            data_dict = json.load(f)
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        
        # If that fails to produce columns, try standard
        if df.shape[1] < 2:
             df = pd.read_json(metadata_path)
             
    except Exception as e:
        logger.warning(f"Initial JSON load failed: {e}. Trying fallback...")
        try:
             df = pd.read_json(metadata_path, lines=True)
        except:
             pass

    if df.empty:
         logger.warning("Loaded DataFrame is empty.")
         return []
         
    logger.info(f"Loaded {len(df)} entries. Columns found: {list(df.columns)[:10]}...")

    # Identify relevant columns
    label_cols = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
        'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
        'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
    ]
    
    # Check if we have labels
    available_labels = [c for c in label_cols if c in df.columns]
    
    # If no labels, we might need to load a separate label file
    if not available_labels:
        logger.warning("No labels found in main metadata. Searching for separate label file...")
        # Look for *label*.csv/json
        label_file = None
        for root, _, files in os.walk(rex_dir):
            for f in files:
                 if 'label' in f.lower() and (f.endswith('.csv') or f.endswith('.json')):
                     label_file = os.path.join(root, f)
                     break
            if label_file: break
        
        if label_file:
            logger.info(f"Found potential label file: {label_file}")
            try:
                if label_file.endswith('.csv'):
                    df_labels = pd.read_csv(label_file)
                else:
                    df_labels = pd.read_json(label_file, orient='index')
                
                # Merge? Or just use this if it has IDs?
                # Usually separate label files have ID column or matches index
                # Let's check overlap
                common_cols = [c for c in label_cols if c in df_labels.columns]
                if common_cols:
                    logger.info(f"Found labels in secondary file: {common_cols}")
                    available_labels = common_cols
                    # We might need to merge, but if images are indexed by filename...
                    # Let's assume this new DF is the master record
                    df = df_labels 
            except Exception as e:
                 logger.error(f"Failed to load label file: {e}")

    # 2. Index Images
    # ReX images are often in subfolders like 'deid_png.part01', etc.
    image_map = {}
    print("Indexing ReX images (this may take a moment)...")
    
    count = 0
    for root, dirs, files in os.walk(rex_dir):
        for f in files:
            if f.endswith('.png') or f.endswith('.jpg'):
                # We map just the filename to path
                image_map[f] = os.path.join(root, f)
                count += 1
                if count % 10000 == 0:
                    print(f"  Indexed {count} images...", end='\r')
    print(f"  Indexed {count} total images.")

    # Text columns for label extraction
    text_cols = [c for c in df.columns if c in ['Findings', 'Impression', 'findings', 'impression', 'text']]
    
    if not text_cols and not available_labels:
        logger.warning(f"Could not identify label OR text columns. Available: {list(df.columns)}")
        return []

    logger.info(f"Extracting labels from text columns: {text_cols}")
    
    # Reuse OpenI mapping logic for text
    # Reuse OpenI mapping logic for text
    mapping = {
        'atelectasis': 'Atelectasis',
        'cardiomegaly': 'Cardiomegaly',
        'effusion': 'Effusion',
        'infiltration': 'Infiltration',
        'mass': 'Mass',
        'nodule': 'Nodule',
        'pneumonia': 'Pneumonia',
        'pneumothorax': 'Pneumothorax',
        'consolidation': 'Consolidation',
        'edema': 'Edema',
        'emphysema': 'Emphysema',
        'fibrosis': 'Fibrosis',
        'thickening': 'Pleural_Thickening',
        'hernia': 'Hernia',
        'no finding': 'No Finding',
        'normal': 'No Finding',
        'clear': 'No Finding',
        'unremarkable': 'No Finding'
    }

    # Data Check
    if len(image_map) < 100:
        logger.warning(f"Only {len(image_map)} images found in ReX directory.")
        has_parts = False
        try:
            if any(f.startswith('deid_png.part') for f in os.listdir(rex_dir)):
                has_parts = True
        except: pass
        if has_parts:
            logger.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logger.error("DETECTED UNEXTRACTED DATASET PARTS (deid_png.partXX)")
            logger.error("You MUST extract these files before running this script.")
            logger.error("Run on HPC: cat deid_png.part* > full.tar.gz && tar -xzvf full.tar.gz")
            logger.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    data = []

    for idx_key, row in tqdm(df.iterrows(), total=len(df), desc="Processing ReX"):
        # Resolve Image Path
        # The key (idx_key) often looks like 'pGRD..._s1.2...'
        # But 'ImagePath' column exists!
        img_path = None
        
        candidates = []
        if 'ImagePath' in row:
            candidates.append(str(row['ImagePath']))
        candidates.append(str(idx_key)) # Try the key itself
        
        for cand in candidates:
             if not cand or str(cand).lower() == 'nan': continue
             
             base = os.path.basename(cand)
             # Try basename
             if base in image_map:
                 img_path = image_map[base]
                 break
             # Try adding png
             if base + ".png" in image_map:
                 img_path = image_map[base + ".png"]
                 break
                 
        if not img_path:
            continue
            
        # Resolve Labels
        label_vec = np.zeros(len(TARGET_PATHOLOGIES), dtype=int)
        
        # Method A: Text Findings (OpenI style)
        if text_cols:
            full_text = " ".join([str(row[c]) for c in text_cols if str(row[c]).lower() != 'nan']).lower()
            
            found = False
            for key, target in mapping.items():
                if key in full_text:
                    if target != 'No Finding':
                        idx = TARGET_PATHOLOGIES.index(target)
                        label_vec[idx] = 1
                        found = True
            
            # If explicit "normal"
            if not found:
                 pass # All zeros = No Finding
                 
        # Method B: One-Hot Columns (if exists)
        elif available_labels:
            for i, target in enumerate(TARGET_PATHOLOGIES):
                if target in available_labels:
                    val = row[target]
                    if val == 1:
                        label_vec[i] = 1
        
        data.append({
            'Image Index': os.path.basename(img_path),
            'Processed Image Path': img_path,
            'Harmonized Labels': ' '.join(map(str, label_vec)),
            'Dataset': 'ReXGradient'
        })
        
    logger.info(f"Loaded {len(data)} valid ReX samples.")
    return data

def generate_complete_dataset():
    DATA_ENTRY_CSV = os.path.join(DATASET_DIR, "Data_Entry_2017.csv")
    
    if not os.path.exists(DATA_ENTRY_CSV):
        logger.error(f"Missing {DATA_ENTRY_CSV}. Cannot process local NIH dataset.")
        return

    # --- Process NIH Locally ---
    logger.info("loading NIH Data Entry CSV...")
    nih_metadata = pd.read_csv(DATA_ENTRY_CSV)
    
    # Map images to local paths
    local_image_map = get_nih_image_paths(DATASET_DIR)
    
    nih_processed = []
    stats = {
        'positive_samples': 0,
        'no_finding': 0,
        'pathology_counts': {p: 0 for p in TARGET_PATHOLOGIES}
    }
    
    for _, row in tqdm(nih_metadata.iterrows(), total=len(nih_metadata), desc="Processing NIH"):
        img_idx = row['Image Index']
        
        if img_idx not in local_image_map:
            continue # Skip missing images
            
        img_path = local_image_map[img_idx]
        raw_labels = row['Finding Labels'].split('|')
        
        label_vec = np.zeros(len(TARGET_PATHOLOGIES), dtype=int)
        has_finding = False
        
        for label in raw_labels:
            if label == 'No Finding':
                stats['no_finding'] += 1
                continue
                
            if label in TARGET_PATHOLOGIES:
                idx = TARGET_PATHOLOGIES.index(label)
                label_vec[idx] = 1
                stats['pathology_counts'][label] += 1
                has_finding = True
        
        if has_finding:
            stats['positive_samples'] += 1
            
        nih_processed.append({
            'Image Index': img_idx,
            'Processed Image Path': img_path,
            'Harmonized Labels': ' '.join(map(str, label_vec)),
            'Dataset': 'NIH'
        })
        
    logger.info(f"Processed {len(nih_processed)} valid NIH samples.")

    # --- Process OpenI ---
    OPENI_REPORTS = os.path.join(DATASET_DIR, 'indiana_reports.csv')
    OPENI_PROJECTIONS = os.path.join(DATASET_DIR, 'indiana_projections.csv')
    OPENI_IMAGES = os.path.join(DATASET_DIR, 'images', 'images_normalized')
    
    openi_data = load_openi_data(OPENI_REPORTS, OPENI_PROJECTIONS, OPENI_IMAGES)
    
    # --- Process ReX / MIMIC ---
    # User has 'cxr_df.csv' directly in Dataset folder, or potentially a ReXGradient subfolder
    rex_data = []
    
    # Priority 1: Check ReXGradient subfolder
    REX_DIR_SUB = os.path.join(DATASET_DIR, 'ReXGradient')
    if os.path.exists(REX_DIR_SUB):
        rex_data = load_rex_data(REX_DIR_SUB)
    
    # Priority 2: Check for cxr_df.csv in root DATASET_DIR if we haven't found data yet
    if not rex_data:
        root_cxr = os.path.join(DATASET_DIR, 'cxr_df.csv')
        if os.path.exists(root_cxr):
            logger.info("Found 'cxr_df.csv' in root dataset directory. Processing as ReX/MIMIC data...")
            # We pass DATASET_DIR as the directory to scan for images, as they might be mixed in
            rex_data = load_rex_data(DATASET_DIR)
            
    # Merge
    all_data = nih_processed + openi_data + rex_data
    df = pd.DataFrame(all_data)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, "train_metadata.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Final Stats
    logger.info("=" * 60)
    logger.info(f"Total Combined Samples: {len(df)}")
    logger.info(f"  NIH: {len(nih_processed)}")
    logger.info(f"  OpenI: {len(openi_data)}")
    logger.info(f"  ReX/MIMIC: {len(rex_data)}")
    logger.info(f"Saved metadata to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Dynamic Dataset Directory Detection works for both Windows and Linux HPC
    # Priority: 1. User's specific HPC path 2. Windows Path 3. Relative Path
    possible_paths = [
        "/home/mpi/Sharon/IIT/Dataset",  # HPC Path
        "D:/IIT/Dataset",                # Local Windows Path
        "../Dataset",                    # Relative Path
        "./data"                         # Default fallback
    ]
    
    DATASET_DIR = None
    for p in possible_paths:
        if os.path.exists(p):
            DATASET_DIR = os.path.abspath(p)
            print(f"Detected dataset directory: {DATASET_DIR}")
            break
            
    if DATASET_DIR is None:
        print("Error: Could not locate Dataset directory. Please check paths.")
        exit(1)

    generate_complete_dataset()




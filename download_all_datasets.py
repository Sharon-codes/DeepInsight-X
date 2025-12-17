"""
Complete Multi-Dataset Downloader for Chest X-Ray AI Training
Downloads and organizes 7 datasets: NIH, OpenI, ReXGradient, CheXpert, MIMIC-CXR, PadChest, VinDr-CXR

Usage:
    python download_all_datasets.py --datasets all
    python download_all_datasets.py --datasets chexpert mimic padchest
    python download_all_datasets.py --list
"""

import os
import argparse
import requests
import subprocess
from pathlib import Path
from tqdm import tqdm
import json

# Base dataset directory
DATASET_DIR = Path("D:/IIT/Dataset")

# Dataset configurations
DATASETS_INFO = {
    'nih': {
        'name': 'NIH ChestX-ray14',
        'size': '~45 GB',
        'images': '112,120',
        'manual': True,
        'url': 'https://nihcc.app.box.com/v/ChestXray-NIHCC',
        'instructions': '''
        NIH ChestX-ray14 must be downloaded manually:
        1. Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC
        2. Download all image ZIP files (images_001.tar.gz through images_012.tar.gz)
        3. Download Data_Entry_2017.csv
        4. Download BBox_List_2017.csv
        5. Extract to: {DATASET_DIR}/images_XXX/images/
        '''
    },
    'openi': {
        'name': 'OpenI (Indiana University)',
        'size': '~7 GB',
        'images': '~7,470',
        'manual': False,
        'auto_download': True,
        'instructions': 'Automatically downloaded during preprocessing'
    },
    'rexgradient': {
        'name': 'ReXGradient-160K',
        'size': '~200 GB',
        'images': '160,000',
        'manual': False,
        'requires_token': True,
        'download_script': 'download_rex_v2.py',
        'instructions': '''
        ReXGradient requires Hugging Face token:
        1. Create account at: https://huggingface.co/
        2. Accept dataset terms: https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K
        3. Get token: https://huggingface.co/settings/tokens
        4. Run: python download_rex_v2.py --token YOUR_TOKEN
        '''
    },
    'chexpert': {
        'name': 'CheXpert (Stanford Hospital)',
        'size': '~440 GB',
        'images': '224,316',
        'manual': True,
        'url': 'https://stanfordmlgroup.github.io/competitions/chexpert/',
        'instructions': '''
        CheXpert requires registration:
        1. Visit: https://stanfordmlgroup.github.io/competitions/chexpert/
        2. Register and request access
        3. Download:
           - CheXpert-v1.0-small.zip (~11 GB, downsampled) OR
           - CheXpert-v1.0.zip (~440 GB, full resolution)
        4. Extract to: {DATASET_DIR}/CheXpert/
        5. Structure:
           {DATASET_DIR}/CheXpert/
           ├── train/
           ├── valid/
           ├── train.csv
           └── valid.csv
        '''
    },
    'mimic': {
        'name': 'MIMIC-CXR (MIT/Beth Israel)',
        'size': '~400 GB',
        'images': '377,110',
        'manual': True,
        'url': 'https://physionet.org/content/mimic-cxr-jpg/2.0.0/',
        'requires_credentialing': True,
        'instructions': '''
        MIMIC-CXR requires PhysioNet credentialing:
        1. Create PhysioNet account: https://physionet.org/register/
        2. Complete CITI training (required for access)
        3. Request access: https://physionet.org/content/mimic-cxr-jpg/2.0.0/
        4. Download using wget (provided after approval):
           wget -r -N -c -np --user USERNAME --ask-password https://physionet.org/files/mimic-cxr-jpg/2.0.0/
        5. Extract to: {DATASET_DIR}/MIMIC-CXR/
        6. Structure:
           {DATASET_DIR}/MIMIC-CXR/
           ├── files/
           ├── mimic-cxr-2.0.0-metadata.csv
           ├── mimic-cxr-2.0.0-chexpert.csv
           └── mimic-cxr-2.0.0-split.csv
        '''
    },
    'padchest': {
        'name': 'PadChest (University of Alicante)',
        'size': '~1 TB',
        'images': '160,868',
        'manual': True,
        'url': 'http://bimcv.cipf.es/bimcv-projects/padchest/',
        'instructions': '''
        PadChest requires registration:
        1. Visit: http://bimcv.cipf.es/bimcv-projects/padchest/
        2. Fill out request form
        3. You will receive download links via email
        4. Download image folders (0-54) and PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
        5. Extract to: {DATASET_DIR}/PadChest/
        6. Structure:
           {DATASET_DIR}/PadChest/
           ├── 0/
           ├── 1/
           ├── ...
           ├── 54/
           └── PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
        '''
    },
    'vindrcxr': {
        'name': 'VinDr-CXR (Vietnam)',
        'size': '~15 GB',
        'images': '18,000',
        'manual': False,
        'url': 'https://physionet.org/content/vindr-cxr/1.0.0/',
        'requires_credentialing': True,
        'auto_download_script': True,
        'instructions': '''
        VinDr-CXR requires PhysioNet credentialing:
        1. Create PhysioNet account: https://physionet.org/register/
        2. Complete CITI training
        3. Request access: https://physionet.org/content/vindr-cxr/1.0.0/
        4. Use automated download script (after approval):
           python download_vindrcxr.py --username YOUR_USERNAME
        5. Structure:
           {DATASET_DIR}/VinDr-CXR/
           ├── train/
           ├── test/
           ├── annotations/
           └── image_labels_train.csv
        '''
    }
}


def print_dataset_info(dataset_key=None):
    """Print information about datasets"""
    if dataset_key:
        info = DATASETS_INFO.get(dataset_key)
        if info:
            print(f"\n{'='*70}")
            print(f"Dataset: {info['name']}")
            print(f"{'='*70}")
            print(f"Size: {info['size']}")
            print(f"Images: {info['images']}")
            print(f"Manual Download: {'Yes' if info['manual'] else 'No'}")
            if info.get('requires_token'):
                print(f"Requires Token: Yes (Hugging Face)")
            if info.get('requires_credentialing'):
                print(f"Requires Credentialing: Yes (PhysioNet CITI)")
            print(f"\nInstructions:")
            print(info['instructions'].format(DATASET_DIR=DATASET_DIR))
        else:
            print(f"Unknown dataset: {dataset_key}")
    else:
        print("\n" + "="*70)
        print("AVAILABLE DATASETS FOR CHEST X-RAY TRAINING")
        print("="*70)
        for key, info in DATASETS_INFO.items():
            status = "✓ Auto" if not info['manual'] else "⚠ Manual"
            print(f"\n[{key.upper()}] {info['name']}")
            print(f"  Status: {status}")
            print(f"  Size: {info['size']} | Images: {info['images']}")
            if info.get('requires_token'):
                print(f"  Requires: Hugging Face token")
            if info.get('requires_credentialing'):
                print(f"  Requires: PhysioNet credentialing (CITI training)")


def check_dataset_exists(dataset_key):
    """Check if dataset is already downloaded"""
    checks = {
        'nih': DATASET_DIR / 'images_001' / 'images',
        'openi': DATASET_DIR / 'indiana_reports.csv',
        'rexgradient': DATASET_DIR / 'ReXGradient',
        'chexpert': DATASET_DIR / 'CheXpert' / 'train.csv',
        'mimic': DATASET_DIR / 'MIMIC-CXR' / 'mimic-cxr-2.0.0-metadata.csv',
        'padchest': DATASET_DIR / 'PadChest' / 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv',
        'vindrcxr': DATASET_DIR / 'VinDr-CXR' / 'image_labels_train.csv'
    }
    
    path = checks.get(dataset_key)
    if path and path.exists():
        return True, path
    return False, path


def download_vindrcxr(username, password=None):
    """Download VinDr-CXR using PhysioNet credentials"""
    print("\n📥 Downloading VinDr-CXR...")
    
    output_dir = DATASET_DIR / 'VinDr-CXR'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = 'https://physionet.org/files/vindr-cxr/1.0.0/'
    
    # If password not provided, prompt for it
    if not password:
        import getpass
        password = getpass.getpass("Enter PhysioNet password: ")
    
    # Download using wget
    cmd = [
        'wget', '-r', '-N', '-c', '-np',
        '--user', username,
        '--password', password,
        '--directory-prefix', str(output_dir),
        base_url
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ VinDr-CXR downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading VinDr-CXR: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have access approved on PhysioNet")
        print("2. Complete CITI training if not done")
        print("3. Check username/password")
        return False
    except FileNotFoundError:
        print("❌ wget not found. Please install wget:")
        print("   Windows: choco install wget")
        print("   Or download from: https://eternallybored.org/misc/wget/")
        return False


def generate_download_report():
    """Generate a report of dataset download status"""
    print("\n" + "="*70)
    print("DATASET DOWNLOAD STATUS REPORT")
    print("="*70)
    
    total_size_gb = 0
    total_images = 0
    downloaded_count = 0
    
    for key, info in DATASETS_INFO.items():
        exists, path = check_dataset_exists(key)
        status = "✅ Downloaded" if exists else "❌ Not Downloaded"
        
        print(f"\n{info['name']}")
        print(f"  Status: {status}")
        print(f"  Location: {path}")
        
        if exists:
            downloaded_count += 1
            # Parse size (rough estimate)
            size_str = info['size'].replace('~', '').replace(' GB', '').replace(' TB', '')
            if 'TB' in info['size']:
                size_gb = float(size_str) * 1000
            else:
                size_gb = float(size_str)
            total_size_gb += size_gb
            
            # Parse image count
            img_count = int(info['images'].replace(',', '').replace('~', ''))
            total_images += img_count
    
    print("\n" + "="*70)
    print(f"Summary: {downloaded_count}/{len(DATASETS_INFO)} datasets downloaded")
    if downloaded_count > 0:
        print(f"Total Size: ~{total_size_gb:.0f} GB")
        print(f"Total Images: ~{total_images:,}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Download and manage chest X-ray datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python download_all_datasets.py --list
  python download_all_datasets.py --info chexpert
  python download_all_datasets.py --check
  python download_all_datasets.py --download vindrcxr --username myuser
        '''
    )
    
    parser.add_argument('--list', action='store_true',
                      help='List all available datasets')
    parser.add_argument('--info', type=str, metavar='DATASET',
                      help='Show detailed info for specific dataset')
    parser.add_argument('--check', action='store_true',
                      help='Check which datasets are already downloaded')
    parser.add_argument('--download', type=str, metavar='DATASET',
                      help='Download specific dataset (only automated ones)')
    parser.add_argument('--username', type=str,
                      help='PhysioNet username (for MIMIC/VinDr)')
    parser.add_argument('--password', type=str,
                      help='PhysioNet password (for MIMIC/VinDr)')
    parser.add_argument('--token', type=str,
                      help='Hugging Face token (for ReXGradient)')
    
    args = parser.parse_args()
    
    # Ensure dataset directory exists
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.list:
        print_dataset_info()
    elif args.info:
        print_dataset_info(args.info)
    elif args.check:
        generate_download_report()
    elif args.download:
        dataset = args.download.lower()
        
        if dataset == 'vindrcxr':
            if not args.username:
                print("❌ Error: --username required for VinDr-CXR")
                return
            download_vindrcxr(args.username, args.password)
        
        elif dataset == 'rexgradient':
            if not args.token:
                print("❌ Error: --token required for ReXGradient")
                print("Get token from: https://huggingface.co/settings/tokens")
                return
            print("\n📥 Downloading ReXGradient...")
            subprocess.run(['python', 'download_rex_v2.py', '--token', args.token])
        
        else:
            info = DATASETS_INFO.get(dataset)
            if not info:
                print(f"❌ Unknown dataset: {dataset}")
                return
            
            if info['manual']:
                print(f"\n⚠️  {info['name']} requires manual download")
                print_dataset_info(dataset)
            else:
                print(f"\n📥 {info['name']} will be downloaded during preprocessing")
                print("Run: python create_full_dataset.py")
    else:
        # Default: show help
        parser.print_help()
        print("\n")
        print_dataset_info()


if __name__ == '__main__':
    main()

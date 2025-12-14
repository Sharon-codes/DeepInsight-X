import os
from huggingface_hub import snapshot_download, login
import argparse

def download_dataset(target_dir, token=None):
    """
    Downloads the ReXGradient dataset to the specified directory.
    """
    dataset_id = "rajpurkarlab/ReXGradient-160K"
    
    print(f"Preparing to download {dataset_id}...")
    print(f"Target Directory: {target_dir}")
    
    # Authenticate if token provided
    if token:
        print("Authenticating with provided token...")
        try:
            login(token=token, add_to_git_credential=False)
        except Exception as e:
            print(f"Login warning (non-fatal): {e}")
    else:
        print("No token provided via args. If the download fails (401), please run 'huggingface-cli login' first or pass --token.")

    try:
        os.makedirs(target_dir, exist_ok=True)
        # Download
        print("Starting download... This may take a while.")
        path = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=target_dir,
            local_dir_use_symlinks=False,  # Download actual files
            resume_download=True
        )
        print(f"Successfully downloaded to {path}")
        
    except Exception as e:
        print(f"\nERROR: Download failed. Reason: {e}")
        print("-" * 50)
        print("TROUBLESHOOTING:")
        print("1. This is a GATED dataset. You must accept terms at: https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K")
        print("2. You must have a Hugging Face Account and Access Token (Read).")
        print("3. Run: python download_rex_v2.py --token YOUR_HF_TOKEN")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ReXGradient Dataset")
    parser.add_argument("--dir", type=str, default=".", help="Target directory (default: current dir)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Access Token")
    
    args = parser.parse_args()
    
    # Determine directory (Generic logic similar to create_dataset)
    if args.dir == ".":
        # Heuristics for HPC vs Local
        possible_paths = [
            "/home/mpi/Sharon/IIT/Dataset/ReXGradient",
            "D:/IIT/Dataset/ReXGradient",
            "../Dataset/ReXGradient",
            "./ReXGradient"
        ]
        target_dir = "./ReXGradient"
        for p in possible_paths:
            # If parent exists, use it
            parent = os.path.dirname(p)
            if os.path.exists(parent):
                target_dir = p
                break
    else:
        target_dir = args.dir

    download_dataset(target_dir, args.token)

import pandas as pd
import numpy as np
import os

def verify_metadata():
    print("Verifying metadata CSVs...")
    
    files = [
        "data/processed/train_metadata.csv",
        "data/processed/val_metadata.csv",
        "data/processed/test_metadata.csv"
    ]
    
    for f in files:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            continue
            
        print(f"\nChecking {f}...")
        df = pd.read_csv(f)
        print(f"  Rows: {len(df)}")
        
        if 'Harmonized Labels' not in df.columns:
            print("  ERROR: 'Harmonized Labels' column missing!")
            continue
            
        # Check first few labels
        print(f"  First 5 labels: {df['Harmonized Labels'].head().tolist()}")
        
        # Check for non-zero labels
        has_findings = False
        zero_count = 0
        
        for label_str in df['Harmonized Labels']:
            # Assuming format "0 0 1 0 ..."
            if '1' in str(label_str):
                has_findings = True
            else:
                zero_count += 1
        
        print(f"  Samples with no findings: {zero_count} ({zero_count/len(df)*100:.2f}%)")
        
        if not has_findings:
            print("  WARNING: No findings (all zeros) detected in this file!")
        else:
            print("  SUCCESS: Found positive labels.")

if __name__ == "__main__":
    verify_metadata()

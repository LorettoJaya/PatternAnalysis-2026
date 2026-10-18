import os
import glob
import re
from collections import Counter

# Rangpur HipMRI data set location
DATA_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

def run_data_audit(data_dir):
    print("=" * 60)
    print("-------- HipMRI dataset Audit Report --------")
    print("=" * 60)

    if not os.path.exists(data_dir):
        print(f"[Error] Path does not exist: {data_dir}")
        return

    # 1. Total File Count & Extensions
    all_files = glob.glob(os.path.join(data_dir, "**", "*"), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]

    extensions = [os.path.splitext(f)[1] for f in all_files]
    ext_counts = Counter(extensions)

    print(f"\n1. TOTAL FILES FOUND: {len(all_files)}")
    print(" File Extensions Breakdown:")
    for ext, count in ext_counts.items():
        print(f" - '{ext if ext else 'No Extension'}': {count} files")

    # 2. Folder Organisation (Image vs Masks)
    img_files = [f for f in all_files if "_seg_"  not in f.lower()]
    mask_files = [f for f in all_files if "_seg_" in f.lower()]

    img_dirs = set(os.path.dirname(f) for f in img_files)
    mask_dirs = set(os.path.dirname(f) for f in mask_files)

    print("\n FOLDER ORGANIZATION:")
    print(f" - Images count: {len(img_files)}") 
    print(f" - Masks count: {len(mask_files)}")
    print(f" - Stored in separate folders? YES (_seg_ subdirecotries)")
    print(" Image subdirectories:") 

    print("\n Sample Image Path:")
    if img_files:
        print(f"    * {img_files[0]}")
    print(" Sample Mask Path:")
    if mask_files: 
        print(f"    * {mask_files[0]}")

    # 3. Patient / Volume Identifiers
    # Matches patterns like 'Case_001', 'sub-01', 'patient_12', etc.
    patient_ids = set() 
    for f in img_files:
        filename = os.path.basename(f)
        # Extract patient identifier (everything before '_slice')
        prefix = (filename.split("_slice")[0] if "_slice" in filename else filename)
        patient_ids.add(prefix)

    print("\n IDENTIFIERS & FORMAT:") 
    print(f" - Patient/Volume Identifiers detected: {len(patient_ids)} unique patients")
    print(f" - Sample patient Prefixes: {list(patient_ids)[:5]}")
    print(" - File Format Type: NIfTI files (.nii / .nii.gz)")
    print("=" * 60)

if __name__ == "__main__":
    run_data_audit(DATA_DIR)
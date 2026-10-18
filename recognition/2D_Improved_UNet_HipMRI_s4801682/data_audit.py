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
    img_files = [f for f in all_files if "_image" in f or "image" in f.lower()]
    mask_files = [f for f in all_files if "_slice" in f or "mask" in f.lower() or "_seg" in f.lower()]

    img_dirs = set(os.path.dirname(f) for f in img_files)
    mask_dirs = set(os.path.dirname(f) for f in mask_files)

    print("n2\. FOLDER ORGANIZATION:")
    print(f" - Images count: {len(img_files)}") 
    print(f" - Masks count: {len(mask_files)}")
    print(f" - Stored in separate folders? {'YES' if img_dirs != mask_dirs else 'NO (Same directory)'}")
    print(" Image subdirectories:") 
    for d in img_dirs: 
        print(f"    * {d}")

    # 3. Patient / Volume Identifiers
    # Matches patterns like 'Case_001', 'sub-01', 'patient_12', etc.
    patient_ids = set() 
    for f in img_files:
        filename = os.path.basename(f)
        match = re.search(r'(case[_\-]?\d+|patient[_\-]?\d+|sub[_\-]?\d+|\d{3,})', filename, re.IGNORECASE)
        if match:
            patient_ids.add(match.group(1))

    print("\\n3\. IDENTIFIERS &amp; FORMAT:") 
    print(f" - Patient/Volume Identifiers detected: {len(patient_ids)} unique patients")
    if patient_ids: 
        print(f" - Sample Patient IDs: {list(patient_ids)[:5]}")
    else:
        print(" - [Note] Patient IDs embedded in slice prefix naming.")

    # 4. Storage Format Verification
    if any(f.lower().endswith((".nii", ".nii.gz")) for f in all_files):
        fmt = "NIfTI files (.nii / .nii.gz)"
    elif any(f.lower().endswith((".png", ".jpg")) for f in all_files):
        fmt = "Individual 2D PNG Image Slices"
    elif any(f.lower().endswith((".npz", ".npy")) for f in all_files):
        fmt = "NumPy Array Files (.npz / .npy)"
    else: 
        fmt = "Other / Mixed Format"

    print(f" - File Format Type: {fmt}") 
    print("=" * 60)

if __name__ == "__main__":
    run_data_audit(DATA_DIR)
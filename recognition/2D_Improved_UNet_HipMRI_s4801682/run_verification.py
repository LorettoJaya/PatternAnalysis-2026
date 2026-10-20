from pathlib import Path 
import re 
import numpy as np 
import matplotlib.pyplot as plt 
import nibabel as nib
from tqdm import tqdm

"""
check enviroment: confirm PyTorch, NiBabel and NumPy are installed and check whether CUDA is avaliable
Inspect the data set: count the NIfTI files and sow the folder organisation
Check image-mask pairs: match filenames using case, week and slice IDs and report missing or duplicate matches. 
inspect sample data: load a few image-mask pairs, check their dimensions, intensity values and mask labels.
check the existing splits: check is the current folder organisation seperates case IDs across training, validation and testing. 
save a visual example: create an image showing an MRI slice with its mask overlay, so you can manually inspect it. 
"""

DATA_DIR = Path("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")

# Regex pattern matching case_XXX_week_Y_slice_Z.nii.gz and seg_XXX_week_y_slice_z.nii.gz
IMAGE_PATTERN = re.compile(
    r"^case_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz$",
    re.IGNORECASE
)

# MASK_PATTERN = re.compile(
#     r"^seg_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz$",
#     re.IGNORECASE
# )

# APPENDIX B LISTING 1: course standard NIfTI loading functions
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """Converts a label array into multi-channel 
       categorical (one-hot) encoding.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels: 
        c = int(c) 
        res[..., c : c + 1][arr == c] = 1 
    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    """Load medical image data from names, cases list provided into a list for each. 
       This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.
       nornImage: bool (normalise the image 0.0-1.0)
       early_stop: Stop loading pre-maturely, leaves arrays mostly empty, 
            for quick loading and testin scripts.      
    """
    affines = []

    # Get fixed size
    num = len(imageNames)

    # Read first case to get fixed 2D dimensions
    first_case = nib.load(str(imageNames[0])).get_fdata(caching='unchanged')

    # Remove extra 3D dimensions if present
    if len(first_case.shape) == 3: 
        first_case = first_case[:, :, 0]

    # Multi-channel categorical mask encoding if requested
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames, desc="Loadin NIfTI Slices")):
        niftiImage = nib.load(str(inName))
        inImage = niftiImage.get_fdata(caching="unchanged") # caching='unchanged' avoids memory leaks
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0] #sometimes extra dims in HipMRI_study data

        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i,:,:] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


#1. Verify Image and Mask Alignment:
def verify_alignment(data_dir: Path):
    print("\n1. Verifying Image & Mask Alignment...")

    # Load sample image slices from training folder
    img_dir = data_dir / "keras_slices_train"
    mask_dir = data_dir / "keras_slices_seg_train"

    # Select one training slice and matching mask 
    sample_img_paths = sorted(list(img_dir.glob("case_*.nii.gz")))[:1]
    
    if not sample_img_paths:
        raise FileNotFoundError(f"No NIfTI images found in {img_dir}")

    # Select 1 image for inspection
    sample_img_path = sample_img_paths[0]

    # Validate the image filename
    image_match = IMAGE_PATTERN.match(sample_img_path.name)

    if image_match is None:
        raise ValueError( 
            f"Unexpected image filename: {sample_img_path.name}"
        )

    # Match the mask using case, week and slice identifiers
    case_id, week_id, slice_id = image_match.groups()

    sample_mask_name = (f"seg_{case_id}_week_{week_id}_slice_{slice_id}.nii.gz")
    sample_mask_path = mask_dir / sample_mask_name

    # Load image and mask using the course standard loader
    raw_img = load_data_2D([sample_img_path], normImage=False)
    norm_img = load_data_2D([sample_img_path], normImage=True)
    mask_data = load_data_2D([sample_mask_path], categorical=False)

    # Extract the 2D slices from the batch dimension
    raw_img = raw_img[0]
    norm_img = norm_img[0]
    mask_data = mask_data[0]

    print(f"    Image Path: {sample_img_path.relative_to(data_dir)}")
    print(f"    * Raw Array Shape: {raw_img.shape} | Intensity Range: [{raw_img.min():.2f}, {raw_img.max():.2f}]")
    print(f"    Mask Path: {sample_mask_path.relative_to(data_dir)}")
    print(f"    * Mask Array Shape: {mask_data.shape} | Unique Label Values: {np.unique(mask_data)}")

    # Check spatial dimensions
    if raw_img.shape != mask_data.shape:
        raise ValueError(
            f"Image and mask dimensions do not match:"
            f"{raw_img.shape} vs {mask_data.shape}"
        )

    print("Image and mask dimensions match.")

    # Generate 3-panel visualisation: MRI slice, Mask, and Overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(norm_img, cmap="gray")
    axes[0].set_title("Input Pelvic MRI Slice (Normalized)")

    axes[1].imshow(mask_data, cmap="gray")
    axes[1].set_title("Segmentation Mask")

    axes[2].imshow(norm_img, cmap="gray")
    axes[2].imshow(np.ma.masked_where(mask_data == 0, mask_data), cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay Spatial Alignment Check")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("alignment_check.png")
    plt.close(fig)
    print("     Saved visual alignment overlay plot to 'alignment_check.png'")

if __name__ == "__main__":
    verify_alignment(DATA_DIR)

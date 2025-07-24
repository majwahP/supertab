"""
Module for generating and refining patch-based trabecular bone masks from MHD files 
and saving them in a Zarr-compatible format.
"""

import zarr
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from skimage.transform import resize
import torch
import sys
import dask.array as da
sys.path.append(str(Path(__file__).resolve().parents[1]))
from supertrab.sr_dataset_utils import create_dataloader
from tqdm import tqdm



def create_and_save_trabecular_mask(file_path, data_dir, group_name, patch_size):
    """
    Loads a trabecular bone mask from an external MHD file, downsamples it to patch resolution,
    and saves it as a binary mask dataset within a Zarr group.

    The function performs the following steps:
    - Loads a 3D binary mask (e.g., trabecular region) from a .mhd file using SimpleITK.
    - Resizes the mask to match a fixed target shape (needs to be a factor of zarr chunk size).
    - Reduces the resolution to patch level by grouping voxel blocks and marking a patch as True 
      if it contains any foreground voxels.
    - Saves the resulting binary patch-level mask to the Zarr dataset under the corresponding group.

    Args:
        file_path (Path or str): Path to the root Zarr file.
        data_dir (Path or str): Path to the external directory containing the MHD mask.
        group_name (str): Name of the Zarr group corresponding to the current subject or scan.
        patch_size (tuple): Size of the patch (Z, Y, X) used for downsampling the mask.

    Returns:
        root (zarr.Group): The opened Zarr root group.
        scan_group (zarr.Group): The Zarr group corresponding to `group_name`.
        trabecular_mask (np.ndarray): The binary mask at patch resolution.
    """
    root = zarr.open(str(file_path))
    scan_group = root[group_name]

    for dataset_name in scan_group:
        if any(dataset_name.endswith(suffix) for suffix in [
            "_bone_mask", "_trabecular_mask", "_trabecular_mask_mean_variance", "_variance", "_mean", "_split", "_volume", "_test", "_test2"
        ]):
            continue

        mhd_mask_path = Path(f"{data_dir}/QCT/QCTFEMUR_2019L/masks/QCTFEMUR_2019L_R_HR_trab.mhd")
        if not mhd_mask_path.exists():
            print(f"Skipping {group_name}, no mask file found.")
            continue

        mhd_mask = sitk.ReadImage(mhd_mask_path)
        import_mask = sitk.GetArrayFromImage(mhd_mask).astype(bool)

        print(f"Original imported mask shape: {import_mask.shape}")

        import_mask_resized = resize(
            import_mask,
            (128, 512, 512),
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(bool)

        print(f"Resized mask shape: {import_mask_resized.shape}")

        mask_shape = tuple(s // p for s, p in zip(import_mask_resized.shape, patch_size))
        print(f"Patch-compressed mask shape: {mask_shape}")

        trabecular_mask = np.zeros(mask_shape, dtype=bool)
        for z in range(mask_shape[0]):
            for y in range(mask_shape[1]):
                for x in range(mask_shape[2]):
                    patch = import_mask_resized[
                        z * patch_size[0] : (z + 1) * patch_size[0],
                        y * patch_size[1] : (y + 1) * patch_size[1],
                        x * patch_size[2] : (x + 1) * patch_size[2]
                    ]
                    if np.any(patch):
                        trabecular_mask[z, y, x] = 1

        scan_group.create_dataset(
            f"{dataset_name}_trabecular_mask",
            data=trabecular_mask.astype(np.int8),
            dtype="i2",
            overwrite=False,
            chunks=(1, mask_shape[-2], mask_shape[-1])
        )
    
    return root, scan_group, trabecular_mask


def filter_mask_by_variance(root, scan_group, trabecular_mask, dataset_name, patch_size, variance_threshold):
    """
    Filters a binary patch-based trabecular mask by evaluating local intensity variance 
    within each patch and keeping only those patches with variance above a given threshold.

    This function uses a dataloader to iterate over image patches defined by the input 
    `trabecular_mask`, calculates the variance of each patch, and creates a new mask 
    that only includes patches considered "informative" (i.e., above the variance threshold).

    Args:
        root (zarr.Group): The opened Zarr root group.
        scan_group (zarr.Group): The specific Zarr group where the dataset is stored.
        trabecular_mask (np.ndarray): A boolean mask at patch resolution, indicating which patches to evaluate.
        dataset_name (str): The base name of the dataset to which the filtered mask will be saved.
        patch_size (tuple): The size of each patch (Z, Y, X).
        variance_threshold (float): Minimum variance required for a patch to be retained.
    """
    file_path = Path(root.store.path)
    dataloader = create_dataloader(file_path, draw_same_chunk=True, patch_size=patch_size, shuffle=False, num_workers=6, isSRDataset=False)

    expected_patches = np.sum(trabecular_mask)
    print(f"Expecting ~{expected_patches} patches")

    image_shape = (5921, 4608, 4608)
    mask_shape = trabecular_mask.shape

    variance_filtered_mask = np.zeros_like(trabecular_mask, dtype=bool)

    for i, batch in enumerate(tqdm(dataloader, desc="Filtering patches by variance")):
        positions = batch[0]
        patch = batch[1]
        variances = torch.var(patch, dim=(1, 2, 3), unbiased=False)

        for idx, (pos, var) in enumerate(zip(positions, variances)):
            z_start, y_start, x_start = [int(p[0].item()) for p in pos]
            z = round(z_start / (image_shape[0] / mask_shape[0]))
            y = round(y_start / (image_shape[1] / mask_shape[1]))
            x = round(x_start / (image_shape[2] / mask_shape[2]))

            if var.item() > variance_threshold:
                variance_filtered_mask[z, y, x] = True

    scan_group.create_dataset(
        f"{dataset_name}_trabecular_mask_variance",
        data=variance_filtered_mask.astype(np.int8),
        dtype="i2",
        overwrite=False,
        chunks=(1, mask_shape[-2], mask_shape[-1])
    )


def save_mhd_image_to_zarr_group(
    mhd_path,
    zarr_root_path,
    group_name,
    dataset_name="registered_LR",
    chunks=(128, 512, 512)
):
    """
    Loads a registered image from a .mhd file, upscales it by a given factor, and saves it into a Zarr group.

    Args:
        mhd_path (str or Path): Path to the .mhd file.
        zarr_root_path (str or Path): Path to the root Zarr group.
        group_name (str): Subgroup name inside the Zarr hierarchy.
        dataset_name (str): Name of the dataset to save in the group.
        chunks (tuple): Chunk size for Dask-backed storage.
        upscale_factor (int): Factor to upscale the image in all dimensions.
    """
    mhd_path = Path(mhd_path)
    zarr_root_path = Path(zarr_root_path)

    print("Reading image")

    # Read image from MHD
    image_sitk = sitk.ReadImage(str(mhd_path)) 
    image_np = sitk.GetArrayFromImage(image_sitk) 
    print(f"Loaded image shape: {image_np.shape}, dtype: {image_np.dtype}")

    print(f"Saving to Zarr at: {zarr_root_path}")
    root = zarr.open(str(zarr_root_path), mode="a")
    group = root.require_group(group_name)

    if dataset_name in group:
        print(f"Overwriting existing dataset: {dataset_name}")
        del group[dataset_name]

    group.create_dataset(
        name=dataset_name,
        data=image_np,
        chunks=chunks,
        dtype=np.float32
    )

    print("Done.")

import zarr
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from skimage.transform import resize
import torch
from create_dataloader import create_dataloader
from tqdm import tqdm


# Define patch size
patch_size = (2, 4, 4) #Define interpoplation size of mask
variance_threshold = 1

# Open .zarr file
file_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
data_dir = "/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/00_resampled_data/2019_L"

root: zarr.hierarchy.Group = zarr.open(str(file_path))

group_name = "2019_L"
scan_group: zarr.hierarchy.Group = root[group_name]
#position_metrics_map = scan_group.attrs["metrics_map"]

for dataset_name in scan_group:
    # Skip creation of folder for already existing masks
    if (
        dataset_name.endswith("_bone_mask")
        or dataset_name.endswith("_trabecular_mask")
        or dataset_name.endswith("_trabecular_mask_mean_variance")
        or dataset_name.endswith("_variance")
        or dataset_name.endswith("_mean")
    ):
        continue
    
    # Load external mask
    mhd_mask_path = Path(f"{data_dir}/QCT/QCTFEMUR_2019L/masks/QCTFEMUR_2019L_R_HR_trab.mhd")
    
    if not mhd_mask_path.exists():
        print(f"Skipping {group_name}, no mask file found.")
        continue
    
    mhd_mask = sitk.ReadImage(mhd_mask_path)
    import_mask = sitk.GetArrayFromImage(mhd_mask)
    import_mask = import_mask.astype(bool)

    print(f"Original imported mask shape: {import_mask.shape}")

    # Resize using nearest neighbor interpolation
    import_mask_resized = resize(
        import_mask,
        (128, 512, 512),
        order=0, #nearest naighbor, copy value from closest original voxel, no smoothing          
        preserve_range=True, #keep values 0 or 1
        anti_aliasing=False #dont smooth edges
    ).astype(bool)

    print(f"Resized mask shape: {import_mask_resized.shape}")

    #find how many patches that fit along one axis
    mask_shape = (
        import_mask_resized.shape[0] // patch_size[0],
        import_mask_resized.shape[1] // patch_size[1],
        import_mask_resized.shape[2] // patch_size[2],
    )

    print(f"Patch-compressed mask shape: {mask_shape}")

    # Create a patch-based mask
    trabecular_mask = np.zeros(mask_shape, dtype=bool)
    
    # reduce resolution of mask, if any voxel is true, ser whole voxel to true
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
    

    # Save to zarr
    # trabecular_mask.astype(np.int8)
    # scan_group.create_dataset(
    #     f"{dataset_name}_trabecular_mask",
    #     data=trabecular_mask,
    #     dtype="i2",
    #     overwrite=True,
    #     chunks=(1, mask_shape[-2], mask_shape[-1])
    # )

    #take patch variance into account

    variance_filtered_mask = np.zeros_like(trabecular_mask, dtype=bool)
    dataloader = create_dataloader(file_path, draw_same_chunk=True, patch_size=patch_size, shuffle=False, num_workers=6, isSRDataset=False)
    expected_patches = np.sum(trabecular_mask)
    print(f"Expecting ~{expected_patches} patches")
    image_shape = (5921, 4608, 4608) 
    mask_shape = (64, 128, 128)
    #print(f"Length of dataset: {len(dataloader.dataset)}")


    for i, batch in enumerate(tqdm(dataloader, desc="Filtering patches by variance")):
        # Unpack batch
        positions = batch[0]  
        patch = batch[1]      

        # Compute variance
        variances = torch.var(patch, dim=(1, 2, 3), unbiased=False)

        for idx, (pos, var) in enumerate(zip(positions, variances)):
            
            z_start, y_start, x_start = [int(p[0].item()) for p in pos]

            z = round(z_start / (image_shape[0] / mask_shape[0]))
            y = round(y_start / (image_shape[1] / mask_shape[1]))
            x = round(x_start / (image_shape[2] / mask_shape[2]))

            if var.item() > variance_threshold:
                variance_filtered_mask[z, y, x] = True



    print("Loop done")
    variance_filtered_mask.astype(np.int8)
    print(variance_filtered_mask.shape)

    # Save to zarr
    scan_group.create_dataset(
        f"{dataset_name}_trabecular_mask_variance",
        data=variance_filtered_mask,
        dtype="i2",
        overwrite=True,
        chunks=(1, mask_shape[-2], mask_shape[-1])
    )

print("done")
print(root.tree())

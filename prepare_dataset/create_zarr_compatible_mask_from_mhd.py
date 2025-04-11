# import numpy as np
# from pathlib import Path
# import zarr
# import zarr.hierarchy
# import tqdm
# import SimpleITK as sitk
# from skimage.transform import resize

# patch_size = (32, 64, 64) 

# ## Open .zarr file
# file_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab_testf32_128x512x512.zarr")
# data_dir = "/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/00_resampled_data/1955_L" # define group

# root: zarr.hierarchy.Group = zarr.open(str(file_path))

# group_name = "1955_L" # define group
# scan_group: zarr.hierarchy.Group = root[group_name]


# for dataset_name in scan_group:
#     #create group of not already exist
#     if (
#         dataset_name.endswith("_bone_mask")
#         or dataset_name.endswith("_trabecular_mask")
#         or dataset_name.endswith("_trabecular_mask_mean_variance")
#         or dataset_name.endswith("_variance")
#         or dataset_name.endswith("_mean")
#     ):
#         continue
#     ## Load mask ##
#     mhd_mask_path = Path(f"{data_dir}/QCT/QCTFEMUR_1955L/masks/QCTFEMUR_1955L_R_HR_trab.mhd") # define group

#     #Check if mask exist
#     if not mhd_mask_path.exists():
#         print(f"Skipping {group_name}, no mask file found.")
#         continue
        
#     mhd_mask = sitk.ReadImage(mhd_mask_path)
#     import_mask = sitk.GetArrayFromImage(mhd_mask) 
#     import_mask = import_mask.astype(bool)
#     import_mask_resized = resize(import_mask, (128, 512, 512), order=0, preserve_range=True, anti_aliasing=False).astype(bool)

#     print(f"{import_mask.shape=}")

#     ## Save mask to group ## 
#     scan_group.create_dataset(
#         f"{dataset_name}_trabecular_mask",
#         data=import_mask_resized,
#         dtype=bool,
#         overwrite=True,
#         chunks=(1, import_mask.shape[-2], import_mask.shape[-1])  
#     )

# print("done")
# print(root.tree())

import zarr
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from skimage.transform import resize

# Define patch size
patch_size = (2, 4, 4) #Define interpoplation size of mask

# Open .zarr file
file_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab_testf32_128x512x512.zarr")
data_dir = "/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/00_resampled_data/1955_L"

root: zarr.hierarchy.Group = zarr.open(str(file_path))

group_name = "1955_L"
scan_group: zarr.hierarchy.Group = root[group_name]

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
    mhd_mask_path = Path(f"{data_dir}/QCT/QCTFEMUR_1955L/masks/QCTFEMUR_1955L_R_HR_trab.mhd")
    
    if not mhd_mask_path.exists():
        print(f"Skipping {group_name}, no mask file found.")
        continue
    
    mhd_mask = sitk.ReadImage(mhd_mask_path)
    import_mask = sitk.GetArrayFromImage(mhd_mask)
    import_mask = import_mask.astype(bool)

    print(f"Original imported mask shape: {import_mask.shape}")

    # Resize using nearest neighbor interpolation carefully
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

    # Create a patch-based mask like first script
    trabecular_mask_patch = np.zeros(mask_shape, dtype=bool)

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
                    trabecular_mask_patch[z, y, x] = 1

    trabecular_mask_patch.astype(np.int8)
    print(trabecular_mask_patch.shape)

    # Save to zarr
    scan_group.create_dataset(
        f"{dataset_name}_trabecular_mask",
        data=trabecular_mask_patch,
        dtype="i2",
        overwrite=True,
        chunks=(1, mask_shape[-2], mask_shape[-1])
    )

print("done")
print(root.tree())

import numpy as np
from pathlib import Path
import zarr
import zarr.hierarchy
import tqdm
import SimpleITK as sitk

patch_size = (1, 32, 32) 

## Open .zarr file
file_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab_small.zarr")
root: zarr.hierarchy.Group = zarr.open(str(file_path))

## iterate over all groups ##
for group_name in tqdm.tqdm(root):
    scan_group: zarr.hierarchy.Group = root[group_name]

    for dataset_name in scan_group:
        #create group of not already exist
        if (
            dataset_name.endswith("_bone_mask")
            or dataset_name.endswith("_trabecular_mask")
            or dataset_name.endswith("_variance")
            or dataset_name.endswith("_mean")
        ):
            continue
        ## Load mask ##
        mhd_mask_path = Path(f"path/to/your_mask.mhd") # TODO change to correct paths

        #Check if mask exist
        if not mhd_mask_path.exists():
            print(f"Skipping {group_name}, no mask file found.")
            continue
        
        mhd_mask = sitk.ReadImage(mhd_mask_path)
        import_mask = sitk.GetArrayFromImage(mhd_mask) 
        import_mask = import_mask.astype(np.uint8)

        ## Save mask to group ## 
        scan_group.create_dataset(
            f"{dataset_name}_trabecular_mask",
            data=import_mask,
            dtype="uint8",
            overwrite=True,
            chunks=patch_size,  
        )
    print("done")
    print(root.tree())
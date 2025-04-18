"""
This script creates masking to select patches containing cortical bone from zarr dataset containing images of hip-bone.
Before running is a .zarr dataset required which wan be created with:https://github.com/gianthk/pyfabric/blob/master/scripts/supertrab_isq_to_zarr_script.py
and metrics mean and varainace calculated which is done with: supertrab_calculate_metrics.py
The patch size defines the area of one patch that is valid or non valid. So one patch is either
true or false (bone/not bone)
code based on: https://github.com/dveni/pneumodomo/blob/main/scripts/add_mask.py
"""


import numpy as np
from pathlib import Path
import zarr
import zarr.hierarchy
import tqdm



#variables
patch_size = (1, 32, 32) #-Change to preferred size
greyvalue_threshold = 3900 


def main():
    #set path to desired .zarr file
    file_path =  Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab_small.zarr")
    root: zarr.hierarchy.Group = zarr.open(str(file_path))
    for group_name in tqdm.tqdm(root):
        scan_group: zarr.hierarchy.Group = root[group_name]
        print(scan_group)
        for dataset_name in scan_group:
            print(dataset_name)
            if (
                dataset_name.endswith("_bone_mask")
                or dataset_name.endswith("_trabecular_mask")
                or dataset_name.endswith("_variance")
                or dataset_name.endswith("_mean")
            ):
                continue
            dataset: zarr.Array = scan_group[dataset_name]
            #calculate how many patches that fits in dataset ex data is (1,512,521) and patch (1,64,64) -> (1,8,8)
            metrics_mask_shape = tuple(
                dim // patch for dim, patch in zip(dataset.shape, patch_size)
            )
            #create a new dataset in the group of type boolean
            mask_dataset = scan_group.create_dataset(
                f"{dataset_name}_bone_mask",
                shape=metrics_mask_shape,
                dtype="uint8",
                overwrite=True,
                chunks=(1, dataset.shape[-2], dataset.shape[-1]),
            )

            #initiate dataset with 0 for varance mask
            cortical_mask = zarr.zeros(metrics_mask_shape, dtype="uint8")

            def get_slice_patched(position: list, patch_size: tuple):
                return (
                    slice(position[0] // patch_size[0], position[1] // patch_size[0]),
                    slice(position[2] // patch_size[1], position[3] // patch_size[1]),
                    slice(position[4] // patch_size[2], position[5] // patch_size[2]),
                )

            # Function to process each patch
            def process_patch(patch_info):
                if (
                    (patch_info["mean"] > greyvalue_threshold) 
                ):
                    # Calculate the position in the compressed array
                    compressed_position = get_slice_patched(
                        patch_info["position"], patch_size
                    )
                    cortical_mask[compressed_position] = 1
            #debug
            if scan_group.attrs:
                print(f"Attributes in {group_name}: {list(scan_group.attrs.keys())}")
            else:
                print(f"No attributes found in {group_name}.")

            position_metrics_map = scan_group.attrs["metrics_map"]

            variance_patch_list = list(
                position_metrics_map.values()
            )  # Convert dict values to list for iteration

            for patch_info in tqdm.tqdm(variance_patch_list, desc="Processing Patches"):
                process_patch(patch_info)
            
            # to add another mask; mask = np.logical_and(cylindrical_mask, variance_mask[:])

            #variance_mask[:] = binary_opening(binary_closing(binary_closing(variance_mask[:])))

            mask_dataset[:] = cortical_mask #change to mask after operations

            print("Mask dtype:", mask_dataset.dtype)  # Should be uint8 or uint16
            print("Mask shape:", mask_dataset.shape)

    print("done")
    print(root.tree())


if __name__ == "__main__":
    main()
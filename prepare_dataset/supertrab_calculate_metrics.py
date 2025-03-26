"""
This script measures variance and mean value of patches in a .zarr dataset and adds the information to the
dataset. 
It requires a .zarr file which wan be created with for example: https://github.com/gianthk/pyfabric/blob/master/scripts/supertrab_isq_to_zarr_script.py
Code is based on: https://github.com/dveni/pneumodomo/blob/main/scripts/add_mask.py 
"""


import zarr
import zarrdataset as zds
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
import torch


def main():
    PATCH_SIZE = (
        1,
        32,
        32,
    )  # Patch size must be a factor of the shape of the image array
    BATCH_SIZE = 64
    NWORKERS = 8

    zarr_patch_sampler = zds.PatchSampler(PATCH_SIZE)

    # Change to path to .zarr dataset
    file_path =  Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab_small.zarr")
    root: zarr.hierarchy.Group = zarr.open(str(file_path))

    for group_name in tqdm(root):
        print(group_name)

        file_specs = zds.ImagesDatasetSpecs(
            filenames=root,
            data_group=f"{group_name}/image", #change to name of folder to image data in group folders
            source_axes="ZYX",
        )

        zarr_dataset = zds.ZarrDataset(
            [file_specs],
            patch_sampler=zarr_patch_sampler,
            return_positions=True,
            shuffle=False,
            progress_bar=False,
            draw_same_chunk=True,
            return_worker_id=False,
        )

        dataloader = DataLoader(
            zarr_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NWORKERS,
            worker_init_fn=zds.zarrdataset_worker_init_fn,
            prefetch_factor=4,
        )

        position_variance_map = {}


        # calculate variance and mean for each patch and save it as attribute in dataset with position
        for i, (position, patch) in enumerate(tqdm(dataloader)):
            patch = patch.float()
            variances = torch.var(patch, dim=(1, 2, 3))
            means = torch.mean(patch, dim=(1, 2, 3))
            for idx, (pos, var, mean) in enumerate(zip(position, variances, means)):
                position_tuple = tuple(
                    pos.flatten().tolist()
                )
                global_idx = i * BATCH_SIZE + idx
                position_variance_map[global_idx] = {
                    "variance": var.item(),
                    "mean": mean.item(),
                    "position": position_tuple,
                    #can here add more metrics
                }

        scan_group = root[group_name]
        scan_group.attrs["variance_map"] = position_variance_map

    print("done")



if __name__ == "__main__":
    main()
import sys
from pathlib import Path
import zarr
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))  # adjust if needed

from supertrab.sr_dataset_utils import create_dataloader
from supertrab.inferance_utils import scale

PATCH_SIZE = (256, 256, 256)
DOWN_FACTOR = 10
BATCH_SIZE = 1

def count_patches_for_part(zarr_path, part):
    zarr_path = Path(zarr_path)
    root = zarr.open(zarr_path, mode="r")

    if part == None:
        group_path = f"2019_L/registered_LR_upscaled_trimmed"
        mask_path = ""
    else: 
        try:
            if part == 0:
                group_path = f"2019_L/registered_LR_upscaled_trimmed_split/part_1"
                mask_path = f"part_1"
            else:
                group_path = f"2019_L/registered_LR_upscaled_trimmed_split/part_2_split/part_{part}"
                mask_path = f"part_2_split/part_{part}"

            original = root[group_path]
            print(f"Found volume for part {part}: shape={original.shape}, chunks={original.chunks}")
        except KeyError:
            print(f"Part {part} not found in Zarr structure — skipping.")
            return

    dataloader = create_dataloader(
        zarr_path=zarr_path,
        patch_size=PATCH_SIZE,
        downsample_factor=DOWN_FACTOR,
        batch_size=BATCH_SIZE,
        draw_same_chunk=True,
        shuffle=False,
        enable_sr_dataset=True,
        num_workers=0,
        prefetch=None,
        data_dim="3d",
        groups_to_use=["2019_L"],
        image_group=group_path.replace("2019_L/", ""),
        mask_base_path="image_trabecular_mask_split",
        mask_group=mask_path
    )

    patch_counter = 0
    # skipped = 0
    for batch in dataloader:

        # AIR value test ------------------------------
        # hr_patch = batch["hr_image"].squeeze().cpu().numpy() * 32768.0
        # hr_patch = scale(hr_patch)

        # air_threshold = 1000
        # air_mask = hr_patch < air_threshold
        # air_percentage = np.sum(air_mask) / hr_patch.size * 100

        # if air_percentage > 5:
        #     skipped += 1
        #     continue 

        #-------------------------------------------------
        patch_counter += batch["hr_image"].shape[0]

    print(f"Part {part}: {patch_counter} patches\n")
    # print(f"Part {part}: {patch_counter} patches (skipped {skipped} patches with >5% air)\n")

def main():
    zarr_path = "/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr"
    for part in range(0, 17):  # 0–16 inclusive
        count_patches_for_part(zarr_path, part)
    # count_patches_for_part(zarr_path, None)

if __name__ == "__main__":
    main()

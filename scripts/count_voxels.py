import sys
from pathlib import Path
import zarr

sys.path.append(str(Path(__file__).resolve().parents[1]))  # adjust if needed

from supertrab.sr_dataset_utils import create_dataloader

PATCH_SIZE = (1, 256, 256)
DOWN_FACTOR = 8
BATCH_SIZE = 1

def count_patches_for_part(zarr_path, part):
    zarr_path = Path(zarr_path)
    root = zarr.open(zarr_path, mode="r")

    if part == None:
        group_path = f"2019_L/image/reassembled_HR"
        mask_path = ""
    else: 
        try:
            if part == 0:
                group_path = f"2019_L/image_split/part_1"
                mask_path = f"part_1"
            else:
                group_path = f"2019_L/image_split/part_2_split/part_{part}"
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
        data_dim="2d",
        groups_to_use=["2019_L"],
        image_group=group_path.replace("2019_L/", ""),
        mask_base_path="image_trabecular_mask_split",
        mask_group=mask_path
    )

    patch_counter = 0
    for batch in dataloader:
        patch_counter += batch["hr_image"].shape[0]

    print(f"Part {part}: {patch_counter} patches\n")

def main():
    zarr_path = "/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr"
    for part in range(0, 17):  # 0–16 inclusive
        count_patches_for_part(zarr_path, part)
    # count_patches_for_part(zarr_path, None)

if __name__ == "__main__":
    main()

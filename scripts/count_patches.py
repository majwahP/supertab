import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # adjust if needed

from supertrab.sr_dataset_utils import create_dataloader


def count_patches(
    zarr_path,
    #part,
    patch_size=(256, 256, 256),
    downsample_factor=4,
    batch_size=32,
    max_batches=None
):
    dataloader = create_dataloader(
        zarr_path=zarr_path,
        patch_size=patch_size,
        downsample_factor=downsample_factor,
        batch_size=batch_size,
        draw_same_chunk=True,
        shuffle=False,
        enable_sr_dataset=True,
        groups_to_use=["2007_L"],    
        # image_group=f"image_split/part_2_split/part_{part}",        
        # mask_base_path="image_trabecular_mask_split",
        # mask_group=f"part_2_split/part_{part}"
        image_group="image",        
        mask_base_path="image_trabecular_mask",
        mask_group=""
    )

    #print(f"part {part}")

    patch_counter = 0
    for i, batch in enumerate(dataloader):
        batch_size_actual = batch["hr_image"].shape[0]
        patch_counter += batch_size_actual

        if i % 32*500 == 0:
            print(f"Total patches processed so far: {patch_counter}")

        if max_batches is not None and i >= max_batches:
            break

    print(f"\nTotal number of patches: {patch_counter}")

if __name__ == "__main__":
    
    # for p in range(1,17):
        count_patches(
            zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr"),
            # part = p,
            patch_size=(1, 256, 256),
            downsample_factor=4,
            batch_size=32,
            max_batches=None,
            
        )

import os
from pathlib import Path
import sys
import torch
import numpy as np
from tqdm import tqdm
import zarr
import time
import pandas as pd


sys.path.append(str(Path(__file__).resolve().parents[1]))

from supertrab.sr_dataset_utils import create_dataloader
from supertrab.inferance_utils import load_model, generate_sr_images, scale
from diffusers import DDPMScheduler


if len(sys.argv) < 2:
    raise ValueError("Usage: python 2Dto3D.py <PART>")

PART = int(sys.argv[1])
ds_factor = int(sys.argv[2])


def main(
    zarr_path,
    weights_path,
    patch_size,
    downsample_factor,
    batch_size,
    sr_dataset_name,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    zarr_path = Path(zarr_path)
    root = zarr.open(zarr_path, mode="a")

    # Original volume information
    if PART == 0:
        original = root[f"2019_L/registered_LR_upscaled_trimmed_split/part_1"]
        volume_shape = original.shape  
        chunk_shape = original.chunks  # must match patch_size
        print(f"Original volume shape: {volume_shape}, chunks: {chunk_shape}")
        print(f"Part_1")
        print(device)
        output_path = f"2019_L/{sr_dataset_name}/part_1"
        image_group_name = f"registered_LR_upscaled_trimmed_split/part_1"
        mask_group_name = f"part_1"
    else:
        original = root[f"2019_L/registered_LR_upscaled_trimmed_split/part_2_split/part_{PART}"]
        volume_shape = original.shape  
        chunk_shape = original.chunks  # must match patch_size
        print(f"Original volume shape: {volume_shape}, chunks: {chunk_shape}")
        print(f"Part_{PART}")
        print(device)
        output_path = f"2019_L/{sr_dataset_name}/part_2_split/part_{PART}"
        image_group_name = f"registered_LR_upscaled_trimmed_split/part_2_split/part_{PART}"
        mask_group_name = f"part_2_split/part_{PART}"


    # Prepare SR output Zarr array
    if output_path in root:
        del root[output_path]

    sr_dataset = root.create_dataset(
        output_path,
        shape=volume_shape,
        chunks=chunk_shape,
        dtype="f4",
        overwrite=False
    )

    # Load model and scheduler
    model = load_model(weights_path, image_size=patch_size[-1], device=device)
    model.eval()
    torch.set_grad_enabled(False)

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Dataloader
    dataloader = create_dataloader(
        zarr_path=zarr_path,
        patch_size=patch_size,
        downsample_factor=downsample_factor,
        batch_size=batch_size,
        draw_same_chunk=True,
        shuffle=False,
        enable_sr_dataset=True,
        num_workers=16,
        prefetch=4,
        groups_to_use=["2019_L"],    
        image_group=image_group_name,        
        mask_base_path="image_trabecular_mask_split",
        mask_group=mask_group_name,
        override_air_values=False,
        with_blur=False,
    )

    print("Starting inference over full volume...")
    patch_counter = 0

    for batch in tqdm(dataloader, desc="Patches"):

        hr_images = batch["hr_image"].to(device)      
        positions = batch["position"]    
        for i in range(hr_images.shape[0]):
            patch = hr_images[i]                    
            patch_scaled = scale(patch * 32768.0)     
            patch_input = patch_scaled / 32768.0
            patch_input = patch_input.unsqueeze(0)  
            sr_patch = generate_sr_images(
                model,
                scheduler,
                patch_input,
                target_size=patch_size[-1],
                device=device
            )[0] 

            z, y, x = positions[i][:, 0].tolist()
            dz, dy, dx = patch_size
            sr_np = sr_patch.cpu().numpy().astype(np.float32)
            sr_dataset[z:z+dz, y:y+dy, x:x+dx] = sr_np
            patch_counter += 1

        #DBG

        # for patch, position in zip(lr_images, batch["position"]):
        #     print(f"position shape: {position.shape}, values: {position}")

        #     z, y, x = position[:, 0].tolist()  # extract start indices
        #     dz, dy, dx = patch_size
        #     sr_np = patch.cpu().numpy().astype(np.float32)
        #     sr_dataset[z:z+dz, y:y+dy, x:x+dx] = sr_np #save to zarr volume
        #     patch_counter += 1
        
        # end_time = time.time()
        # elapsed = end_time - mid_time
        # print(f"{elapsed} time for saving to volume", flush=True)
        print(f"{patch_counter} patches processed", flush=True)

    # Save excluded positions to CSV
    # project_root = Path(__file__).resolve().parents[1]
    # csv_output_dir = project_root / "air_patches_positions"
    # csv_output_dir.mkdir(exist_ok=True)

    # exclude_csv_path = csv_output_dir / f"excluded_positions_part{PART}_ds{downsample_factor}.csv"
    # df_exclude = pd.DataFrame(positions_to_exclude, columns=["z", "y", "x"])
    # df_exclude.to_csv(exclude_csv_path, index=False)

    # print(f"\nExcluded patch positions saved to: {exclude_csv_path}")

    print(f"\nSuper-resolved volume saved at: {output_path}")
    print(f"\nTotal patches super-resolved and written: {patch_counter}")

if __name__ == "__main__":

    PATCH_SIZE = 256
    DS_FACTOR = ds_factor

    main(
        zarr_path="/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr",
        weights_path=f"samples/supertrab-diffusion-sr-2d-v5/{PATCH_SIZE}_ds{DS_FACTOR}/models/final_model_weights_{PATCH_SIZE}_ds{DS_FACTOR}.pth",
        patch_size=(1, PATCH_SIZE, PATCH_SIZE),
        downsample_factor=DS_FACTOR,
        batch_size=16, 
        sr_dataset_name=f"sr_volume_{PATCH_SIZE}_{ds_factor}_200ep_given_QCT"
    )

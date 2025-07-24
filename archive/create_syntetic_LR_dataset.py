import os
from pathlib import Path
import sys
import torch
import numpy as np
from tqdm import tqdm
import zarr
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))

from supertrab.sr_dataset_utils import create_dataloader
from supertrab.inferance_utils import load_model, generate_dps_sr_images, generate_sr_images
from diffusers import DDPMScheduler


if len(sys.argv) < 2:
    raise ValueError("Usage: python 2Dto3D.py <PART>")

PART = int(sys.argv[1])
ds_factor = int(sys.argv[2])

GROUP_NAME = sys.argv[3]

def inverse_scale(scaled_image):
    image = (scaled_image - 1983.3156) / 2.5902297
    return image


def main(
    zarr_path,
    # weights_path,
    patch_size,
    downsample_factor,
    batch_size,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    zarr_path = Path(zarr_path)
    root = zarr.open(zarr_path, mode="a")
    output_group = f"{GROUP_NAME}_LR/image_splitted"

    # Original volume information
    if PART == 0:
        original = root[f"{GROUP_NAME}/image_split/part_1"]
        volume_shape = original.shape  
        chunk_shape = original.chunks  # must match patch_size
        print(f"Original volume shape: {volume_shape}, chunks: {chunk_shape}")
        print(f"Part_1")
        print(device)
        output_path = f"{output_group}/part_1"
        image_group_name = f"image_split/part_1"
        mask_group_name = f"part_1"
    else:
        original = root[f"{GROUP_NAME}/image_split/part_2_split/part_{PART}"]
        volume_shape = original.shape  
        chunk_shape = original.chunks  # must match patch_size
        print(f"Original volume shape: {volume_shape}, chunks: {chunk_shape}")
        print(f"Part_{PART}")
        print(device)
        output_path = f"{output_group}/part_2_split/part_{PART}"
        image_group_name = f"image_split/part_2_split/part_{PART}"
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
    # model = load_model(weights_path, image_size=patch_size[-1], device=device)
    # model.eval()
    # torch.set_grad_enabled(False)

    # scheduler = DDPMScheduler(num_train_timesteps=1000)

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
        groups_to_use=[GROUP_NAME],    
        image_group=image_group_name,        
        mask_base_path="image_trabecular_mask_split",
        mask_group=mask_group_name, 
        with_blur=True
    )

    print("Starting inference over full volume...")
    patch_counter = 0

    for batch in tqdm(dataloader, desc="Patches"):
        # print("Starting batch")
        # start_time = time.time()
        lr_images = batch["lr_image"].to(device)
        lr_images = inverse_scale(lr_images*32768.0)
        # sr_images = generate_sr_images(
        #     model,
        #     scheduler,
        #     lr_images,
        #     target_size=patch_size[-1],
        #     device=device
        # )

        # mid_time = time.time()
        # elapsed = mid_time - start_time
        # print(f"{elapsed} time for inferance", flush=True)
        for lr_patch, position in zip(lr_images, batch["position"]):
            z, y, x = position[:, 0].tolist()  # extract start indices
            dz, dy, dx = patch_size
            sr_np = lr_patch.cpu().numpy().astype(np.float32)
            sr_dataset[z:z+dz, y:y+dy, x:x+dx] = sr_np #save to zarr volume
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


    print(f"\nSuper-resolved volume saved at: {output_path}")
    print(f"\nTotal patches super-resolved and written: {patch_counter}")

if __name__ == "__main__":

    PATCH_SIZE = 256
    DS_FACTOR = ds_factor

    main(
        # zarr_path="/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr",
        zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr"),
        # weights_path=f"samples/supertrab-diffusion-sr-2d-v5/{PATCH_SIZE}_ds{DS_FACTOR}/models/final_model_weights_{PATCH_SIZE}_ds{DS_FACTOR}.pth",
        patch_size=(1, PATCH_SIZE, PATCH_SIZE),
        downsample_factor=DS_FACTOR,
        batch_size=16, 
    )

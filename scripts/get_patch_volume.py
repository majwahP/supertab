import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from supertrab.sr_dataset_utils import create_dataloader


PATCH_SIZE = 256
DS_FACTOR = 6

def main(
    zarr_path,
    dim,
    batch_size,
    output_dir,
    groups_to_use,
    patch_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
    downsample_factor=DS_FACTOR,
    
):
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting one HR patch...")

    dataloader_HR_LR = create_dataloader(
        zarr_path=zarr_path,
        patch_size=patch_size,
        downsample_factor=downsample_factor,
        groups_to_use=groups_to_use,
        batch_size=batch_size,
        draw_same_chunk=False,
        shuffle=True,
        enable_sr_dataset=True, 
        data_dim=dim, 
        num_workers=0, 
        prefetch=None,
        image_group="image", 
        mask_base_path="image_trabecular_mask",
        mask_group=""
    )
    print("Dataloader created")
    for batch_idx, batch in enumerate(dataloader_HR_LR):
        hr_images = batch["hr_image"]  # shape: (B, D, H, W) or (B, 1, D, H, W)
        
        for i in range(hr_images.shape[0]):
            hr_image_np = hr_images[i].cpu().numpy().astype(np.float32)
            hr_image_np = np.squeeze(hr_image_np)

            sitk_img = sitk.GetImageFromArray(hr_image_np)
            output_path = os.path.join(output_dir, f"hr_patch_{batch_idx}.mhd")
            sitk.WriteImage(sitk_img, output_path)
            print(f"Saved HR patch to {output_path}")

            # Visualization of one example per batch (optional)
            if i == 0:  # or add `and batch_idx == 0` to only do the very first
                axial = hr_image_np[hr_image_np.shape[0] // 2, :, :]
                coronal = hr_image_np[:, hr_image_np.shape[1] // 2, :]
                sagittal = hr_image_np[:, :, hr_image_np.shape[2] // 2]

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                for ax, img, title in zip(axes, [axial, coronal, sagittal], ['Axial', 'Coronal', 'Sagittal']):
                    ax.imshow(img, cmap='gray')
                    ax.set_title(title)
                    ax.axis('off')

                vis_path = os.path.join(output_dir, f"hr_patch_vis_{batch_idx}.png")
                plt.tight_layout()
                plt.savefig(vis_path, dpi=300)
                plt.close()
                print(f"Saved visualization to {vis_path}")
        
        # Optional: stop after N batches
        if batch_idx >= 5:
            break  # Remove or change to collect more


if __name__ == "__main__":
    main(
        zarr_path=Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr"),
        dim = "3d",
        patch_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
        downsample_factor=DS_FACTOR,
        output_dir="patch_outputs",
        groups_to_use=["2019_L"],
        batch_size=1
    )
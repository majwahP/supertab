import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from supertrab.sr_dataset_utils import create_dataloader

PATCH_SIZE = 256
DS_FACTOR = 4

def has_empty_slice(volume: torch.Tensor) -> bool:
    return torch.any(torch.all(volume == 0, dim=(1, 2)))

def normalize_patch(patch):
    patch = patch.cpu().numpy().astype(np.float32)
    patch -= patch.min()
    patch /= (patch.max() + 1e-8)
    return patch

def visualize_patch_grid(hr_list, lr_list, sr_list, save_path="patch_grid.png"):
    assert len(hr_list) == len(lr_list) == len(sr_list), "Mismatch in number of patches"
    num_samples = len(hr_list)

    # Check dimensionality (2D or 3D)
    sample_shape = hr_list[0].shape
    is_3d = len(sample_shape) == 3 and min(sample_shape) > 1

    if is_3d:
        # 3 orthogonal views per patch (HR, LR, SR)
        fig, axes = plt.subplots(num_samples, 9, figsize=(18, 2 * num_samples))
        for i in range(num_samples):
            for j, (patch, label) in enumerate(zip(
                [hr_list[i], lr_list[i], sr_list[i]], 
                ['HR', 'LR', 'SR']
            )):
                volume = normalize_patch(patch)
                axial = volume[volume.shape[0] // 2, :, :]
                coronal = volume[:, volume.shape[1] // 2, :]
                sagittal = volume[:, :, volume.shape[2] // 2]

                for k, view in enumerate([axial, coronal, sagittal]):
                    ax = axes[i, j * 3 + k] if num_samples > 1 else axes[j * 3 + k]
                    ax.imshow(view, cmap='gray')
                    if i == 0:
                        ax.set_title(f"{label} - {['Axial', 'Coronal', 'Sagittal'][k]}", fontsize=8)
                    ax.axis('off')

    else:
        fig, axes = plt.subplots(num_samples, 3, figsize=(6, num_samples * 2))
        for i in range(num_samples):
            for j, (patch, label) in enumerate(zip(
                [hr_list[i], lr_list[i], sr_list[i]], 
                ['HR', 'LR', 'SR']
            )):
                ax = axes[i, j] if num_samples > 1 else axes[j]
                ax.imshow(normalize_patch(patch), cmap='gray')
                if i == 0:
                    ax.set_title(label, fontsize=10)
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved patch grid to {save_path}")


def main(
    zarr_path,
    data_dim,
    patch_size=(1, PATCH_SIZE, PATCH_SIZE),
    downsample_factor=DS_FACTOR,
    batch_size=8,
    output_dir="patch_outputs",
    groups_to_use=["2019_L"],
    device="cuda" if torch.cuda.is_available() else "cpu",
    nr_samples = 20,
):
    os.makedirs(output_dir, exist_ok=True)

    dataloader_HR_LR = create_dataloader(
        zarr_path=zarr_path,
        patch_size=patch_size,
        downsample_factor=downsample_factor,
        groups_to_use=groups_to_use,
        batch_size=batch_size,
        draw_same_chunk=True,
        shuffle=False,
        enable_sr_dataset=True, 
        data_dim=data_dim, 
        num_workers=0, 
        prefetch=None,
        image_group="image_split/reassembled_HR", 
        mask_base_path="image_trabecular_mask_split/reassembled", 
        mask_group=""
    )

    dataloader_SR = create_dataloader(
        zarr_path=zarr_path,
        patch_size=patch_size,
        downsample_factor=downsample_factor,
        groups_to_use=groups_to_use,
        batch_size=batch_size,
        draw_same_chunk=True,
        shuffle=False,
        enable_sr_dataset=True, 
        data_dim=data_dim, 
        num_workers=0, 
        prefetch=None,
        image_group=f"sr_volume_256_{DS_FACTOR}/reassembled",        
        mask_base_path="image_trabecular_mask_split/reassembled",
        mask_group=""
    )

    saved_hr, saved_lr, saved_sr = [], [], []
    total_patches = 0
    sample_idx = 0

    for batch_HR_LR, batch_SR in tqdm(zip(dataloader_HR_LR, dataloader_SR), desc="Collecting patches"):
        lr_images = batch_HR_LR["lr_image"].to(device)
        hr_images = batch_HR_LR["hr_image"].to(device)
        sr_images = batch_SR["hr_image"].to(device) * 32768.0

        for hr_patch, lr_patch, sr_patch in zip(hr_images, lr_images, sr_images):
            if sample_idx % 10 == 0:

                sr = sr_patch[0].cpu()
                hr = hr_patch[0].cpu()
                lr = lr_patch[0].cpu()

                if sr.sum() == 0 or has_empty_slice(sr):
                    continue

                saved_hr.append(hr)
                saved_lr.append(lr)
                saved_sr.append(sr)

                total_patches += 1
                if total_patches >= nr_samples:
                    break

            sample_idx += 1
        if total_patches >= nr_samples:
            break

    print(f"Collected {total_patches} patches. Saving grid...")
    save_path = os.path.join(output_dir, f"patch_grid_ds{DS_FACTOR}_{data_dim}.png")
    visualize_patch_grid(saved_hr, saved_lr, saved_sr, save_path=save_path)

if __name__ == "__main__":
    main(
        zarr_path=Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr"),
        patch_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
        data_dim="3d",
        downsample_factor=DS_FACTOR,
        output_dir="patch_outputs",
        groups_to_use=["2019_L"],
        nr_samples=10
    )

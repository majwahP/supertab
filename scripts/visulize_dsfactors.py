import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from supertrab.sr_dataset_utils import create_dataloader
from supertrab.analysis_utils import has_empty_slice

PATCH_SIZE = 256
DS_FACTORS = [4, 6, 8, 10]
NR_SAMPLES_PER_ROW = 5  # number of patches per DS factor (and HR)


def normalize_patch(patch):
    patch = patch.cpu().numpy().astype(np.float32)
    patch -= patch.min()
    patch /= (patch.max() + 1e-8)
    return patch


def visualize_hr_lr_grid(hr_list, lr_dict, save_path="hr_lr_patch_grid.png"):
    ds_labels = ['HR'] + [f'DS={ds}' for ds in sorted(lr_dict.keys())]
    num_rows = len(ds_labels)
    num_patches = len(hr_list)
    num_views = 3  # axial, coronal, sagittal
    num_cols = num_patches * num_views

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows))

    # Header row: view labels
    for p in range(num_patches):
        for v in range(num_views):
            col_idx = p * num_views + v
            axes[0, col_idx].set_title(["Axial", "Coronal", "Sagittal"][v], fontsize=10)

    # Plot HR first
    for patch_idx, patch_3d in enumerate(hr_list):
        volume = normalize_patch(patch_3d)
        z_mid, y_mid, x_mid = [s // 2 for s in volume.shape]
        slices = [
            volume[z_mid, :, :],  # axial
            volume[:, y_mid, :],  # coronal
            volume[:, :, x_mid],  # sagittal
        ]
        for view_idx, slice_2d in enumerate(slices):
            col_idx = patch_idx * num_views + view_idx
            ax = axes[0, col_idx]
            ax.imshow(slice_2d, cmap="gray")
            ax.axis("off")
            if view_idx == 0:
                ax.set_ylabel("HR", fontsize=10, rotation=0, labelpad=20)

    # Plot LR for each DS factor
    for row_idx, ds in enumerate(sorted(lr_dict.keys()), start=1):
        for patch_idx, patch_3d in enumerate(lr_dict[ds]):
            volume = normalize_patch(patch_3d)
            z_mid, y_mid, x_mid = [s // 2 for s in volume.shape]
            slices = [
                volume[z_mid, :, :],  # axial
                volume[:, y_mid, :],  # coronal
                volume[:, :, x_mid],  # sagittal
            ]
            for view_idx, slice_2d in enumerate(slices):
                col_idx = patch_idx * num_views + view_idx
                ax = axes[row_idx, col_idx]
                ax.imshow(slice_2d, cmap="gray")
                ax.axis("off")
                if view_idx == 0:
                    ax.set_ylabel(f"DS={ds}", fontsize=10, rotation=0, labelpad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved patch grid to {save_path}")


def collect_patches(ds_factor, zarr_path, patch_size, data_dim, groups_to_use, device, nr_samples, is_hr=False):
    dataloader = create_dataloader(
        zarr_path=zarr_path,
        patch_size=patch_size,
        downsample_factor=ds_factor if not is_hr else 1,
        groups_to_use=groups_to_use,
        batch_size=8,
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

    patches = []
    total_collected = 0
    sample_idx = 0

    for batch in tqdm(dataloader, desc=f"{'HR' if is_hr else f'DS={ds_factor}'}"):
        image_key = "hr_image" if is_hr else "lr_image"
        imgs = batch[image_key].to(device)

        for patch in imgs:
            patch = patch[0].cpu()
            if patch.sum() == 0 or has_empty_slice(patch):
                continue
            if sample_idx % 10 == 0:
                patches.append(patch)
                total_collected += 1
            sample_idx += 1
            if total_collected >= nr_samples:
                break
        if total_collected >= nr_samples:
            break

    return patches


def main():
    output_dir = "patch_outputs"
    os.makedirs(output_dir, exist_ok=True)

    zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
    patch_size = (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
    data_dim = "3d"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    groups_to_use = ["2019_L"]

    # Collect HR
    hr_patches = collect_patches(
        ds_factor=1,
        zarr_path=zarr_path,
        patch_size=patch_size,
        data_dim=data_dim,
        groups_to_use=groups_to_use,
        device=device,
        nr_samples=NR_SAMPLES_PER_ROW,
        is_hr=True
    )

    # Collect LR for all DS factors
    lr_dict = {}
    for ds in DS_FACTORS:
        patches = collect_patches(
            ds_factor=ds,
            zarr_path=zarr_path,
            patch_size=patch_size,
            data_dim=data_dim,
            groups_to_use=groups_to_use,
            device=device,
            nr_samples=NR_SAMPLES_PER_ROW,
            is_hr=False
        )
        lr_dict[ds] = patches

    # Visualize
    save_path = os.path.join(output_dir, "hr_lr_patch_grid_3views.png")
    visualize_hr_lr_grid(hr_patches, lr_dict, save_path=save_path)


if __name__ == "__main__":
    main()

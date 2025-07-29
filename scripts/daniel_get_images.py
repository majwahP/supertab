from pathlib import Path
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from supertrab.metrics_utils import get_mask_ormir

# --- Settings ---
patch_size = 256
ds_factors = [4, 6, 8]#, 10]
z, y, x = 128, 1280, 2304
half = patch_size // 2

zarr_root = Path("/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr")
group_name = "2019_L"
output_dir = Path("daniel_images")
output_dir.mkdir(exist_ok=True)
# --- Plot layout: 2 DSFs per row, 3 slices per DSF ---
cols_per_dsf = 3
dsfs_per_row = 2
n_rows = (len(ds_factors) + dsfs_per_row - 1) // dsfs_per_row

fig_img, axes_img = plt.subplots(n_rows, dsfs_per_row * cols_per_dsf, figsize=(20, 4 * n_rows))
fig_mask, axes_mask = plt.subplots(n_rows, dsfs_per_row * cols_per_dsf, figsize=(20, 4 * n_rows))

axes_img = np.array(axes_img).reshape(n_rows, dsfs_per_row * cols_per_dsf)
axes_mask = np.array(axes_mask).reshape(n_rows, dsfs_per_row * cols_per_dsf)

for idx, dsf in enumerate(ds_factors):
    print(f"Processing DSF {dsf}...")

    # Paths
    if dsf == 4:
        path_13 = f"sr_volume_256_{dsf}_200ep/part_2_split/part_13"
        path_14 = f"sr_volume_256_{dsf}_200ep/part_2_split/part_14"
    else:
        path_13 = f"sr_volume_256_{dsf}/part_2_split/part_13"
        path_14 = f"sr_volume_256_{dsf}/part_2_split/part_14"

    # Load and extract patch
    vol_13 = da.from_zarr(zarr_root / group_name / path_13)
    vol_14 = da.from_zarr(zarr_root / group_name / path_14)
    vol = da.concatenate([vol_13, vol_14], axis=0)
    patch = vol[z - half:z + half, y:y + patch_size, x:x + patch_size].compute()

    # Get mask
    patch_tensor = torch.from_numpy(patch).unsqueeze(0)
    mask = get_mask_ormir(patch_tensor)  # shape (D, H, W)

    # Extract orthogonal slices
    axial_img = patch[patch_size // 2, :, :]
    coronal_img = patch[:, patch_size // 2, :]
    sagittal_img = patch[:, :, patch_size // 2]

    axial_mask = mask[patch_size // 2, :, :].numpy()
    coronal_mask = mask[:, patch_size // 2, :].numpy()
    sagittal_mask = mask[:, :, patch_size // 2].numpy()

    row = idx // dsfs_per_row
    col_start = (idx % dsfs_per_row) * cols_per_dsf

    for j, (img_slice, mask_slice, label) in enumerate(zip(
        [axial_img, coronal_img, sagittal_img],
        [axial_mask, coronal_mask, sagittal_mask],
        ["Axial", "Coronal", "Sagittal"]
    )):
        ax_img = axes_img[row, col_start + j]
        ax_mask = axes_mask[row, col_start + j]

        ax_img.imshow(img_slice, cmap="gray")
        ax_img.set_title(f"DSF {dsf} - {label}", fontsize=14)
        ax_img.axis("off")
        ax_img.set_aspect("equal")

        ax_mask.imshow(mask_slice, cmap="gray")
        ax_mask.set_title(f"DSF {dsf} - {label} Mask", fontsize=14)
        ax_mask.axis("off")
        ax_mask.set_aspect("equal")

# --- Save both figures ---
fig_img.tight_layout()
fig_mask.tight_layout()

fig_img.savefig(output_dir / "orthogonal_grid_images.png", dpi=300, bbox_inches="tight")
fig_mask.savefig(output_dir / "orthogonal_grid_masks.png", dpi=300, bbox_inches="tight")

plt.close(fig_img)
plt.close(fig_mask)

print("✅ Saved both image and mask grids to 'daniel_images/'")
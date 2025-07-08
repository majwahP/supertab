"""
This module provides tools for evaluating and debugging patch-based datasets adjusted for 
superresolution. Dataloader can be created by create_dataloader in supertrab.sr_dataset_utils
"""

import torch
import random
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import gc

def has_empty_slice(volume: torch.Tensor) -> bool:
    return torch.any(torch.all(volume == 0, dim=(1, 2)))


def check_patch_uniqueness(dataloader):
    """
    Checks whether any patch positions occur more than once in a PyTorch DataLoader 
    reated with create_dataloader in supertrab.sr_dataset_utils and verifies if 
    repeated patches at the same position are identical.

    Example usecase: ensuring that each patch is sampled only once 
    (e.g., when draw_same_chunk=True).

    Args:
        dataloader (torch.utils.data.DataLoader): A DataLoader yielding batches 
            containing a 'position' key and 'hr_image' key.

    Prints:
        - The total number of patches checked.
        - A list of duplicated patch positions, if any.
        - For each duplicate, whether the associated patches are identical or not.
    """

    position_to_patches = defaultdict(list)
    total_checked = 0

    for batch in tqdm(dataloader, desc="Checking patch positions"):
        positions = batch["position"]
        hr_images = batch["hr_image"]  # High-resolution patches

        for pos, patch in zip(positions, hr_images):
            # Make position hashable
            hashable_pos = tuple((int(dim[0]), int(dim[1])) for dim in pos)

            position_to_patches[hashable_pos].append(patch)
            total_checked += 1

    print(f"Checked {total_checked} patches.\n")

    # Find duplicates
    duplicates = {pos: patches for pos, patches in position_to_patches.items() if len(patches) > 1}

    if len(duplicates) == 0:
        print("All patch positions are unique!")
    else:
        print(f"Found {len(duplicates)} positions that occur multiple times:\n")
        for pos, patches in sorted(duplicates.items(), key=lambda x: -len(x[1])):
            print(f"Position {pos}: {len(patches)} patches")

            
            # Compare all patches at this position
            identical = all(torch.equal(p1, p2) for i, p1 in enumerate(patches) for p2 in patches[i+1:]) # compare every patch to every other patch

            if identical:
                print(f"    -> Patches at {pos} are IDENTICAL")
            else:
                print(f"    -> Patches at {pos} are DIFFERENT")

def visualize_masks(hr_list, lr_list, sr_list, save_path="mask_grid.png"):
    assert len(hr_list) == len(lr_list) == len(sr_list), "Mismatch in number of masks"

    num_samples = len(hr_list)
    fig, axes = plt.subplots(num_samples, 3, figsize=(6, num_samples * 2))

    for i in range(num_samples):
        for j, (mask, label) in enumerate(zip(
            [hr_list[i], lr_list[i], sr_list[i]], 
            ['HR', 'LR', 'SR']
        )):
            ax = axes[i, j] if num_samples > 1 else axes[j]
            ax.imshow(mask.cpu().numpy(), cmap='gray')
            if i == 0:
                ax.set_title(label, fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure to {save_path}")

def visualize_3d_masks(hr_list, lr_list, sr_list, save_path="mask_grid_3d.png"):
    """
    For each sample, show HR, LR, SR in a row with 3 orthogonal slices (axial, coronal, sagittal)
    """
    num_samples = len(hr_list)

    fig, axes = plt.subplots(num_samples, 9, figsize=(9 * 2, num_samples * 2))

    for i in range(num_samples):
        masks = {
            'HR': hr_list[i].cpu().numpy(),
            'LR': lr_list[i].cpu().numpy(),
            'SR': sr_list[i].cpu().numpy()
        }

        for j, (label, volume) in enumerate(masks.items()):
            mid_slices = [
                volume[volume.shape[0] // 2, :, :],  # axial (Z)
                volume[:, volume.shape[1] // 2, :],  # coronal (Y)
                volume[:, :, volume.shape[2] // 2],  # sagittal (X)
            ]
            for k, slice_2d in enumerate(mid_slices):
                ax = axes[i, j * 3 + k] if num_samples > 1 else axes[j * 3 + k]
                ax.imshow(slice_2d, cmap='gray')
                if i == 0:
                    ax.set_title(f"{label} - {['Axial', 'Coronal', 'Sagittal'][k]}", fontsize=8)
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved 3D orthogonal slice grid to {save_path}")

def plot_random_samples_from_dataloader(dataloader, output_path="samples.png", max_samples=50):
    """
    Randomly selects and visualizes HR/LR image patch pairs from a PyTorch DataLoader, 
    arranging them in a grid and saving the result as a PNG image.

    Each sample is displayed with the low-resolution (LR) patch on the left and the 
    corresponding high-resolution (HR) patch on the right, making it easy to inspect 
    the quality and alignment of the generated data.

    Args:
        dataloader (torch.utils.data.DataLoader): A DataLoader yielding batches that include 
            'hr_image' and 'lr_image' tensors of shape [B, 1, H, W].
        output_path (str): File path where the resulting image grid will be saved.
        max_samples (int): Maximum number of HR/LR pairs to visualize.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid_cols = 5
    grid_rows = (max_samples + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
    axes = axes.flatten()

    collected = 0
    for batch in tqdm(dataloader, desc="Collecting samples for plot"):
        batch_hr = batch["hr_image"]
        batch_lr = batch["lr_image"]

        batch_size = batch_hr.shape[0]
        indices = random.sample(range(batch_size), min(batch_size, max_samples - collected))

        for idx in indices:
            if collected >= max_samples:
                break

            hr = batch_hr[idx]
            lr = batch_lr[idx]
            pair = torch.cat([lr, hr], dim=-1)  # concat left-right

            ax = axes[collected]
            ax.imshow(pair[0].cpu().numpy(), cmap="gray")
            ax.axis("off")
            collected += 1

        if collected >= max_samples:
            break

    # Hide any unused subplots
    for idx in range(collected, len(axes)):
        axes[idx].axis('off')
    print("Saving image")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)  # Important to free memory!
    gc.collect()


def visualize_orthogonal_slices(volume, title="Orthogonal Slices", save_path=None, cmap='gray'):
    """
    Visualize the three orthogonal mid-slices (axial, coronal, sagittal) of a 3D volume.

    Args:
        volume (torch.Tensor or np.ndarray): 3D volume (D, H, W)
        title (str): Title for the entire plot
        save_path (str): If given, saves the figure to this path
        cmap (str): Matplotlib colormap (default: 'gray')
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()

    axial    = volume[volume.shape[0] // 2, :, :]
    coronal  = volume[:, volume.shape[1] // 2, :]
    sagittal = volume[:, :, volume.shape[2] // 2]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    views = [axial, coronal, sagittal]
    labels = ["Axial", "Coronal", "Sagittal"]

    for ax, img, label in zip(axes, views, labels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(label)
        ax.axis('off')

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")
    plt.show()

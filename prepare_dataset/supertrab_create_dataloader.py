"""
This script creates a PyTorch dataloader from a zarr dataset containing images of hip-bone.
Before running is a .zarr dataset required which wan be created with:https://github.com/gianthk/pyfabric/blob/master/scripts/supertrab_isq_to_zarr_script.py
and a mask to from which areas the samples should be collected needs to be defined which can be 
done with for example: supertrab_create_mask_bone.py or supertrab_create_mask_trabecular_bone.py.
The patch size defines the area of a sample. This script also collects a 20 samples from the 
dataloader and plots them and their pisitions in the image in two seperate plots saved in a new 
folder named images.
this code is based on: https://github.com/dveni/pneumodomo/blob/main/scripts/lung_dataloader.py
"""

from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import zarr
import zarr.hierarchy
import zarrdataset as zds
from tqdm import tqdm
import matplotlib.patches as patches
import shutil

from torch.utils.data import DataLoader


def visualize_group(group, group_name, patch_sampler, output_dir, max_samples=20):
    # Dataset specs
    file_specs = zds.ImagesDatasetSpecs(filenames=[group], data_group="image", source_axes="ZYX")
    masks_specs = zds.MasksDatasetSpecs(filenames=[group], data_group="image_trabecular_mask", source_axes="ZYX")

    # Dataset and loader
    dataset = zds.ZarrDataset(
        [file_specs, masks_specs],
        patch_sampler=patch_sampler,
        return_positions=True,
        shuffle=True,
        progress_bar=True,
        draw_same_chunk=False,
        return_worker_id=False,
    )

    dl = DataLoader(
        dataset,
        batch_size=30,
        num_workers=10,
        worker_init_fn=zds.zarrdataset_worker_init_fn,
        prefetch_factor=2,
    )

    # Sample patches
    samples = []
    positions = []
    for _, (pos, sample) in enumerate(tqdm(dl, desc=f"Sampling {group_name}")):
        for p in pos:
            positions.append(p)
            if len(positions) >= max_samples:
                break
        samples.append(sample)
        if len(samples) >= max_samples:
            break

    grid_samples = np.vstack(samples)

    # Save patch grid
    plt.figure(figsize=(20, 20))
    for i, image in enumerate(grid_samples[:max_samples]):
        plt.subplot(4, 5, i + 1)
        plt.imshow(image[0], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"samples_{group_name}.png")
    plt.close()

    # Save positions
    image = group["image"]
    fig, axes = plt.subplots(4, 5, figsize=(15, 12), dpi=500)
    axes = axes.flatten()

    for idx, (m_z, m_y, m_x) in enumerate(positions[:max_samples]):
        ax = axes[idx]
        ax.imshow(image[m_z[0].item(), :, :], cmap="gray")
        rect = patches.Rectangle(
            (m_x[0], m_y[0]),
            m_x[1] - m_x[0],
            m_y[1] - m_y[0],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    for idx in range(max_samples, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / f"positions_{group_name}.png")
    plt.close()


def main():
    PATCH_SIZE = (1, 128, 128)
    ZARR_PATH = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab_small.zarr")
    OUTPUT_DIR = Path("images")
    OUTPUT_DIR.mkdir(exist_ok=True)

    patch_sampler = zds.PatchSampler(PATCH_SIZE, min_area=0.999)

    root: zarr.hierarchy.Group = zarr.open(str(ZARR_PATH))
    named_groups = root.groups()
    
    for name, group in named_groups:
        visualize_group(group, name, patch_sampler, OUTPUT_DIR)


if __name__ == "__main__":
    main()
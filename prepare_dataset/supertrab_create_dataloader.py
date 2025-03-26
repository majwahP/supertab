from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import zarr
import zarr.hierarchy
import zarrdataset as zds
from tqdm import tqdm
import matplotlib.patches as patches

from torch.utils.data import DataLoader


def main():
    PATCH_SIZE = 1, 128, 128
    zarr_patch_sampler = zds.PatchSampler(PATCH_SIZE, min_area=0.25)
    file_path =  Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab_small.zarr")
    
    root: zarr.hierarchy.Group = zarr.open(str(file_path))
    groups = [g[1] for g in root.groups()][:1] #g[0] is group name, remove [:1] to use all images

    file_specs = zds.ImagesDatasetSpecs(
        filenames=groups,
        data_group="image",
        source_axes="ZYX",
    )
    masks_specs = zds.MasksDatasetSpecs(
        filenames=groups,
        data_group="image_bone_mask",
        source_axes="ZYX",
    )

    zarr_dataset = zds.ZarrDataset(
        [file_specs, masks_specs],
        patch_sampler=zarr_patch_sampler,
        return_positions=True,
        shuffle=True,
        progress_bar=True,
        draw_same_chunk=False,
        return_worker_id=False,
    )

    #create dataloader
    dl = DataLoader(
        zarr_dataset,
        batch_size=30,
        num_workers=20,
        worker_init_fn=zds.zarrdataset_worker_init_fn,
        prefetch_factor=2,
    )





    #visulize samples from dataloader

    max_samples = 10
    samples = []
    positions = []
    for i, (pos, sample) in enumerate(tqdm(dl)):
        positions.append(pos)
        samples.append(sample)
        if len(samples) >= max_samples:
            break   

    grid_samples = np.vstack(samples)

    #plot patches in grid
    os.makedirs("images", exist_ok=True)
    plt.figure(figsize=(20, 10))
    for i, image in enumerate(grid_samples[:10]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(image[0], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("images/samples.png")

    image = root["1955_L/image"]

    #prepare plot
    n_plots = len(positions)  # Maximum 10 plots
    n_cols = 5  # You can adjust this
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division to get required rows

    # Create one figure with multiple subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    # Loop through the positions
    for idx, (m_z, m_y, m_x) in enumerate(positions[0]):
        if idx >= 10:
            break

        #print(m_z, m_y, m_x)

        # Plot on the corresponding subplot
        ax = axes[idx]
        ax.imshow(image[m_z[0].item(), :, :], cmap="gray")

        # Create and add the rectangle
        width = m_x[1] - m_x[0]
        height = m_y[1] - m_y[0]
        rect = patches.Rectangle(
            (m_x[0], m_y[0]),
            width,
            height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    # Hide any empty subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    # Adjust the layout and save
    plt.tight_layout()
    plt.savefig("images/all_positions.png")
    plt.close()

    #end saving samples



if __name__ == "__main__":
    main()
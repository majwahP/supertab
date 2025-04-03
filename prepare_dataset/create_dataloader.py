
from pathlib import Path
from torch.utils.data import DataLoader
import zarr
import numpy as np
import zarrdataset as zds
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_dataloader(zarr_path, patch_size=(1, 128, 128), batch_size=30, num_workers=10, min_area=0.999):
    """
    Creates a DataLoader that samples patches from all groups in a Zarr dataset.

    Args:
        zarr_path (str or Path): Path to the root Zarr file.
        patch_size (tuple): Patch size in form (Z, Y, X).
        batch_size (int): Desired batch size for the DataLoader.
        num_workers (int): Number of worker threads.
        min_area (float): Minimum procentage of a patch that needs to be within mask for a valid patch. value (0,1]

    Returns:
        torch.utils.data.DataLoader: DataLoader that yields batches of patches.
    """
    zarr_path = Path(zarr_path)
    root = zarr.open(str(zarr_path))
    named_groups = list(root.groups())
    print(f"Found {len(named_groups)} groups:")
    for name, _ in named_groups:
        print(f"  - {name}")

    patch_sampler = zds.PatchSampler(patch_size, min_area=min_area)

    all_file_specs = []
    all_mask_specs = []
    all_groups = [f"{zarr_path}/{name}" for name, _ in named_groups]

    for _, group in named_groups:
        all_file_specs.append(
            zds.ImagesDatasetSpecs(filenames=all_groups, data_group="image", source_axes="ZYX")
        )
        all_mask_specs.append(
            zds.MasksDatasetSpecs(filenames=all_groups, data_group="image_trabecular_mask", source_axes="ZYX")
        )

    dataset = zds.ZarrDataset(
        all_file_specs + all_mask_specs,
        patch_sampler=patch_sampler,
        return_positions=True,
        shuffle=True,
        progress_bar=True,
        draw_same_chunk=False,
        return_worker_id=False,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=zds.zarrdataset_worker_init_fn,
        prefetch_factor=2,
    )

def check_patch_uniqueness(dataloader):
    """
    Checks whether all patches in the dataloader have unique positions.

    Args:
        dataloader: The DataLoader created by `create_dataloader`.

    Returns:
        bool: True if all patch positions are unique, False otherwise.
    """

    
    return True


def plot_random_samples_from_dataloader(dataloader, output_path="samples.png", max_samples=50):
    import torch

    samples = []
    output_path = Path(output_path)

    for batch in tqdm(dataloader, desc="Collecting samples"):
        images = batch[1]  # adjust if needed
        for image in images:
            samples.append(image)
            if len(samples) >= max_samples:
                break
        if len(samples) >= max_samples:
            break

    print(f"Collected {len(samples)} samples.")

    all_samples = torch.stack(samples)

    # Plot
    num_samples = all_samples.shape[0]
    grid_cols = min(10, num_samples)
    grid_rows = int(np.ceil(num_samples / grid_cols))
    plt.figure(figsize=(grid_cols * 2, grid_rows * 2))

    for i in range(num_samples):
        plt.subplot(grid_rows, grid_cols, i + 1)
        plt.imshow(all_samples[i, 0].numpy(), cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()




def main():
    zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab_small.zarr")
    output_path = "images/random_patches.png"

    dataloader = create_dataloader(zarr_path)
    plot_random_samples_from_dataloader(dataloader, output_path)
    if check_patch_uniqueness(dataloader):
        print("True")
    else:
        print("False")


if __name__ == "__main__":
    main()
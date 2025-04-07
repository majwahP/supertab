
from pathlib import Path
from torch.utils.data import DataLoader
import zarr
import numpy as np
import zarrdataset as zds
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from collections import defaultdict

def get_superresolution_batch_fn(sigma, downsample_factor):
    """
    Creates a batch function that generates low-resolution (LR) images 
    from high-resolution (HR) patches by applying Gaussian blur and downsampling.

    Args:
        sigma (float): Standard deviation for the Gaussian blur applied before downsampling.
        downsample_factor (int): Factor by which to downsample the HR images to create LR images.

    Returns:
        Callable: A collate function that takes a list of samples and returns a batch dictionary with:
            - 'position': Tensor of patch positions
            - 'hr_image': Tensor of high-resolution patches
            - 'lr_image': Tensor of corresponding low-resolution patches
    """
    gaussian_blur = T.GaussianBlur(kernel_size=5, sigma=sigma)

    def LR_batch_fn(batch):
        positions, hr_images = [], []

        for sample in batch:
            position = torch.from_numpy(sample[0])  # First = position
            hr_image = torch.from_numpy(sample[1])  # Second = HR image
            
            positions.append(position)
            hr_images.append(hr_image.float())
        
        lr_images = []

        for hr_img in hr_images:
            smoothed = gaussian_blur(hr_img).unsqueeze(0) #for correct interpolation size
            #Downsample
            lr_img_down = F.interpolate(
                smoothed,
                scale_factor=1 / downsample_factor,
                mode="bilinear",
                align_corners=False
            )

            # Upsample back to original HR size
            lr_img_up = F.interpolate(
                lr_img_down,
                size=hr_img.shape[-2:],  # Match original HR size
                mode="bilinear",
                align_corners=False
            )
            lr_img_up = lr_img_up.squeeze(0)
            lr_images.append(lr_img_up)

        return {
            "position": torch.stack(positions),
            "hr_image": torch.stack(hr_images),
            "lr_image": torch.stack(lr_images),
        }

    return LR_batch_fn


def create_dataloader(zarr_path, patch_size=(1, 128, 128), batch_size=30, num_workers=10, min_area=0.999, sigma=1.3, downsample_factor=4):
    """
    Creates a DataLoader that samples patches from all groups in a Zarr dataset. The dataloader returns both 
    high resolution patches and corresponding low resolution (downsampled) patches.

    Args:
        zarr_path (str or Path): Path to the root Zarr file.
        patch_size (tuple): Patch size in form (Z, Y, X).
        batch_size (int): Desired batch size for the DataLoader.
        num_workers (int): Number of worker threads.
        min_area (float): Minimum procentage of a patch that needs to be within mask for a valid patch. value (0,1]
        sigma (float): Standard deviation for Gaussian blur applied before downsampling.
        downsample_factor (int): Factor by which to downsample the HR patches to create LR patches.
    Returns:
        torch.utils.data.DataLoader: A DataLoader that yields batches containing:
        - 'position': Tensor with patch positions
        - 'hr_image': Tensor with high-resolution patches
        - 'lr_image': Tensor with corresponding low-resolution patches
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

    #debug, try with first group
    name, group = named_groups[0]
    first_group_path = f"{zarr_path}/{name}"

    all_file_specs.append(
        zds.ImagesDatasetSpecs(filenames=[first_group_path], data_group="image", source_axes="ZYX")
    )
    all_mask_specs.append(
        zds.MasksDatasetSpecs(filenames=[first_group_path], data_group="image_trabecular_mask", source_axes="ZYX")
    )

    #for _, group in named_groups:
    #    all_file_specs.append(
    #        zds.ImagesDatasetSpecs(filenames=all_groups, data_group="image", source_axes="ZYX")
    #    )
    #    all_mask_specs.append(
    #        zds.MasksDatasetSpecs(filenames=all_groups, data_group="image_trabecular_mask", source_axes="ZYX")
    #    )

    dataset = zds.ZarrDataset(
        all_file_specs + all_mask_specs,
        patch_sampler=patch_sampler,
        return_positions=True,
        shuffle=True,
        progress_bar=True,
        draw_same_chunk=False,
        return_worker_id=False,
    )

    LR_batch_fn = get_superresolution_batch_fn(sigma=sigma, downsample_factor=downsample_factor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=zds.zarrdataset_worker_init_fn,
        collate_fn = LR_batch_fn,
        prefetch_factor=2,
    )

def check_patch_uniqueness(dataloader):
    """
    Checks for duplicate patch positions in the dataloader.

    Args:
        dataloader: The DataLoader created by `create_dataloader`.

    Prints:
        Each position that occurs more than once, and whether the patches are identical.
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



def plot_random_samples_from_dataloader(dataloader, output_path="samples.png", max_samples=50):

    hr_samples, lr_samples = [], []
    output_path = Path(output_path)

    for batch in tqdm(dataloader, desc="Collecting samples"):
        hr_images = batch["hr_image"]
        lr_images = batch["lr_image"]

        for hr, lr in zip(hr_images, lr_images):
            hr_samples.append(hr)
            lr_samples.append(lr)
            if len(hr_samples) >= max_samples:
                break
        if len(hr_samples) >= max_samples:
            break

    print(f"Collected {len(hr_samples)} HR/LR pairs.")

    # Concatenate HR and upsampled LR images horizontally for each sample
    pairs = [torch.cat([lr, hr], dim=-1) for lr, hr in zip(lr_samples, hr_samples)]
    all_pairs = torch.stack(pairs)  # Shape: [B, 1, H, 2*W]

    # Plot in a grid
    num_samples = all_pairs.shape[0]
    grid_cols = min(5, num_samples)
    grid_rows = int(np.ceil(num_samples / grid_cols))
    plt.figure(figsize=(grid_cols * 4, grid_rows * 4))

    for i in range(num_samples):
        plt.subplot(grid_rows, grid_cols, i + 1)
        plt.imshow(all_pairs[i, 0].numpy(), cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()




def main():
    zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
    output_path = "images/random_patches.png"

    dataloader = create_dataloader(zarr_path)
    plot_random_samples_from_dataloader(dataloader, output_path)
    check_patch_uniqueness(dataloader)


if __name__ == "__main__":
    main()
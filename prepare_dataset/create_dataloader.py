"""
Write a description, code inspired by  https://github.com/dveni/pneumodomo/blob/main/scripts/lung_dataloader.py
the way to create LR pairs is based on: https://gist.github.com/dveni/8bab9b25f7e9c0873a8ef360c45b27e9
"""



import torch
import torch.nn.functional as F
import torchvision.transforms as T
import zarrdataset as zds
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import zarr
import gc

class ImageSample:
    """
    A helper class to manage sampling patches within a chunk.
    
    Attributes:
        im_id (int): Image ID.
        chk_id (int): Chunk ID.
        shuffle (bool): Whether to shuffle patch sampling.
    """
    _current_patch_idx = 0
    _ordering = None
    _rng_seed = None
    num_patches = None

    def __init__(self, im_id: int, chk_id: int, shuffle: bool = False):
        self.im_id = im_id
        self.chk_id = chk_id
        self._shuffle = shuffle

        if self._shuffle:
            self._rng_seed = random.randint(1, 100000)

    def free_sampler(self):
        """Free the memory used for the current patch ordering."""
        del self._ordering
        self._ordering = None

    def next_patch(self):
        """
        Get the next patch index to sample.

        Returns:
            (int, bool): (patch_index, is_empty)
        """
        if self._shuffle and self._ordering is None:
            curr_state = random.getstate()
            random.seed(self._rng_seed)
            self._ordering = list(range(self.num_patches))
            random.shuffle(self._ordering)
            random.setstate(curr_state)

        if self._shuffle:
            curr_patch = self._ordering[self._current_patch_idx]
        else:
            curr_patch = self._current_patch_idx

        self._current_patch_idx += 1
        is_empty = self._current_patch_idx >= self.num_patches

        return curr_patch, is_empty


class SRDataset(zds.ZarrDataset):
    def __init__(self, sigma=1.3, downsample_factor=4, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.downsample_factor = downsample_factor
        self.gaussian_blur = T.GaussianBlur(kernel_size=5, sigma=sigma)
    
    def generate_lr(self, hr_patch):
        #gaussian blurr, downsampling and resize
        
        # Add channel dimension: [1, H, W] for blurr function to work on 2D
        hr_patch = hr_patch.unsqueeze(0).unsqueeze(0)
        smoothed = self.gaussian_blur(hr_patch)

        lr_downsampled = F.interpolate(
            smoothed,
            scale_factor=1/self.downsample_factor,
            mode="bilinear",
            align_corners=False
        )
        lr_resized = F.interpolate(
            lr_downsampled,
            size=hr_patch.shape[-2:],  # Restore original HR shape
            mode="bilinear",
            align_corners=False
        )
        return lr_resized.squeeze(0).squeeze(0)

    def __iter__(self):
        print("Get samples")
        self._initialize()

        print("Get samples 2")
        # Create a list of samples togheter with witch image and chunk id they correspond to, shuffle chnks to get random chunk
        samples = [
            ImageSample(im_id, chk_id, shuffle=self._shuffle)
            for im_id in range(len(self._arr_lists))
            for chk_id in range(len(self._toplefts[im_id]))
        ]

        print("Shuffle")
        #randomly shuffle the samples if shuffeling is enabeled and we want samples from same chunk untill we have all samples from that chunk
        if self._shuffle and self._draw_same_chunk:
            random.shuffle(samples)
        
        #initilize -1 (not loaded yet)
        prev_im_id = -1
        prev_chk_id = -1
        prev_chk = -1
        #First sample
        current_chk = 0
        #No image data loaded yet, loaded to memory when needed
        self._curr_collection = None

        print("While samples loop")
        while samples:
            print("in loop")
            #draw a random chunk from samples if not want from same untill have gotten all
            if self._shuffle and not self._draw_same_chunk:
                current_chk = random.randrange(0, len(samples))
            
            im_id = samples[current_chk].im_id
            chk_id = samples[current_chk].chk_id
            position = self._toplefts[im_id][chk_id]

            if prev_im_id != im_id or chk_id != prev_chk_id: #if have gotten a new chunk
                #if alsready had a previous chunk, free memory
                if prev_chk >= 0:
                    samples[prev_chk].free_sampler()

                #update to current chunk
                prev_chk = current_chk
                prev_chk_id = chk_id

                #update if have sample from new image, load the new image
                if prev_im_id != im_id:
                    prev_im_id = im_id
                    self._curr_collection = self._arr_lists[im_id]

                if self._patch_sampler is not None:
                    patches_position = self._patch_sampler.compute_patches(self._curr_collection, position)
                else:
                    patches_position = [position]

                #set how many available patches
                samples[current_chk].num_patches = len(patches_position)

                # if no valid patches, remove chunk from samples and reset tracking (no acitve chunk anymore)
                if not patches_position:
                    samples.pop(current_chk)
                    prev_chk = -1
                    continue #go to next sample

            #get indes the patch and if there is more patches in the chunk after this one
            current_patch, is_empty = samples[current_chk].next_patch()

            # if all patches in the chunk has been samples, remove chunk from samples list
            if is_empty:
                samples.pop(current_chk)
                prev_chk = -1

            #load the patch from zarr file
            patch_position = patches_position[current_patch]
            patches = self.__getitem__(patch_position)[0]

            #only grayscale images
            if patches.shape[0] > 1:
                raise ValueError("Only single channel images are supported")

            #to torch tensor
            hr_patch = torch.from_numpy(patches[0]).float()

            # Normalize HR patch to [0, 1]
            hr_patch = hr_patch / 65535.0

            # Create LR patch
            lr_patch = self.generate_lr(hr_patch)

            example = {
                "hr_image": hr_patch.unsqueeze(0),  # [1, H, W]
                "lr_image": lr_patch.unsqueeze(0),  # [1, H, W]
            }

            #find positon    
            if self._return_positions:
                pos = [
                    [patch_position[ax].start if patch_position[ax].start is not None else 0,
                     patch_position[ax].stop if patch_position[ax].stop is not None else -1]
                    if ax in patch_position else [0, -1]
                    for ax in self._collections[self._ref_mod][0]["axes"]
                ]
                example["position"] = torch.tensor(pos, dtype=torch.int64)

            if self._return_worker_id:
                example["wid"] = torch.tensor(self._worker_id, dtype=torch.int64)

            print("Have an example")
            yield example


def create_dataloader(zarr_path, patch_size=(1, 128, 128), batch_size=4, num_workers=1, min_area=0.5, sigma=1.3, downsample_factor=4):
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

    #patch_sampler = zds.PatchSampler(patch_size, min_area=min_area)
    patch_sampler = None

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

    dataset = SRDataset(
        dataset_specs=all_file_specs + all_mask_specs,
        patch_sampler= patch_sampler,
        return_positions=True,
        shuffle=True,
        progress_bar=True,
        draw_same_chunk=False,
        return_worker_id=False,
        sigma=sigma,
        downsample_factor=downsample_factor,
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



def plot_random_samples_from_dataloader(dataloader, output_path="samples.png", max_samples=10):
    """
    Plots random HR/LR patch pairs from a dataloader and saves them in a grid.

    Args:
        dataloader: PyTorch DataLoader yielding batches with 'hr_image' and 'lr_image'.
        output_path (str): Where to save the figure.
        max_samples (int): How many total samples to plot.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid_cols = 5
    grid_rows = (max_samples + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
    axes = axes.flatten()

    collected = 0
    print("Collecting batches")
    for batch in tqdm(dataloader, desc="Collecting samples for plot"):
        print("next batch")
        batch_hr = batch["hr_image"]
        batch_lr = batch["lr_image"]

        batch_size = batch_hr.shape[0]
        indices = random.sample(range(batch_size), min(batch_size, max_samples - collected))

        for idx in indices:
            print("Collected: ")
            print(collected)
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






def main():
    zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
    output_path = "images/random_patches.png"

    dataloader = create_dataloader(zarr_path)
    print("done grreating dataloader")
    print("Plotting samples")
    plot_random_samples_from_dataloader(dataloader, output_path)
    print("Check uniqueness")
    check_patch_uniqueness(dataloader)
    print("Done!")


if __name__ == "__main__":
    main()
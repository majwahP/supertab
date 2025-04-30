"""
This module provides custom dataset and dataloader functionality for patch-based 
image super-resolution tasks using Zarr-formatted volumetric data.

Classes and Functions:

- ImageSample:
    A helper class for managing the order and sampling of patches within image chunks, 
    with optional shuffling.

- SRDataset:
    A subclass of ZarrDataset that yields high-resolution (HR) patches and 
    generates corresponding low-resolution (LR) patches on-the-fly using Gaussian 
    blur and bilinear downsampling. It supports both random patch sampling and 
    exhaustive sampling per chunk.

- create_dataloader():
    Constructs a PyTorch DataLoader that iterates over HR/LR patch pairs from 
    all specified groups in a Zarr dataset. The patches are sampled according 
    to a user-defined patch size and filtering criteria, and can be shuffled 
    or iterated exhaustively. Supports both SR-specific and standard dataset modes.

"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import zarrdataset as zds
import random
from pathlib import Path
from torch.utils.data import DataLoader
import zarr


class ImageSample:
    """
    Helper class for managing patch sampling within a specific image chunk.

    This class keeps track of which patch to sample next, optionally shuffles the 
    sampling order (with reproducibility), and provides a clean interface for 
    iterative access to patches in a chunk.

    Attributes:
        im_id (int): Index of the image in the dataset.
        chk_id (int): Index of the chunk within the image.
        shuffle (bool): Whether to shuffle the patch sampling order.
        num_patches (int): Total number of patches available in the chunk.
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
    """
    A custom PyTorch-compatible dataset for super-resolution tasks using Zarr-stored 
    volumetric data.

    SRDataset extends ZarrDataset to generate paired high-resolution (HR) and low-resolution 
    (LR) image patches. HR patches are extracted directly from the dataset, normalized, and 
    optionally blurred and downsampled to create corresponding LR patches.

    Args:
    sigma (float): Standard deviation used for Gaussian blur.
    downsample_factor (int): Factor by which to downsample HR patches to obtain LR.
    **kwargs: Additional arguments passed to the base ZarrDataset class.
    """
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
        self._initialize()
        # Create a list of samples togheter with witch image and chunk id they correspond to, shuffle chnks to get random chunk
        samples = [
            ImageSample(im_id, chk_id, shuffle=self._shuffle)
            for im_id in range(len(self._arr_lists))
            for chk_id in range(len(self._toplefts[im_id]))
        ]
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

        while samples:
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

            yield example


def create_dataloader(zarr_path, patch_size=(1, 128, 128), batch_size=4, num_workers=1, min_area=0.999, sigma=1.3, downsample_factor=4, draw_same_chunk = False, shuffle = True, enable_sr_dataset = True, groups_to_use=None):
    """
    Creates a PyTorch DataLoader that samples patch pairs (HR and LR) from all groups 
    in a Zarr dataset for use in super-resolution tasks.

    This function builds a dataset that extracts high-resolution (HR) patches and 
    generates corresponding low-resolution (LR) versions using Gaussian blur and 
    bilinear downsampling. It supports patch filtering based on foreground mask coverage 
    and allows for both random and exhaustive patch sampling strategies.

    Args:
        zarr_path (str or Path): Path to the root Zarr file containing image and mask data.
        patch_size (tuple): Desired patch size as (Z, Y, X).
        batch_size (int): Number of patch pairs per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        min_area (float): Minimum fraction (0, 1] of a patch that must be within the mask 
                        to be considered valid.
        sigma (float): Standard deviation for Gaussian blur applied before downsampling.
        downsample_factor (int): Factor by which to downsample the HR patch to create the LR patch.
        draw_same_chunk (bool): If True, all valid patches from each chunk are returned once; 
                                if False, sampling is random across the entire dataset.
        shuffle (bool): Whether to shuffle patch sampling order.
        enable_sr_dataset (bool): Whether to use SRDataset (with HR/LR generation) or a standard ZarrDataset.
        groups_to_use (list or None): Optional list of group names to restrict sampling to specific subsets.

    Returns:
        torch.utils.data.DataLoader: A DataLoader yielding batches of dictionaries, each containing:
            - 'position': Tensor with patch position information.
            - 'hr_image': Tensor containing the high-resolution patch [1, H, W].
            - 'lr_image': Tensor containing the corresponding low-resolution patch [1, H, W].
    """
    zarr_path = Path(zarr_path)
    root = zarr.open(str(zarr_path))
    named_groups = list(root.groups()) 
    if groups_to_use is not None:
        named_groups = [(name, group) for name, group in named_groups if name in groups_to_use]
    print(f"Found {len(named_groups)} groups:")
    for name, _ in named_groups:
        print(f"  - {name}")

    patch_sampler = zds.PatchSampler(patch_size, min_area=min_area)

    all_file_specs = []
    all_mask_specs = []

    for name, _ in named_groups:
        group_path = f"{zarr_path}/{name}"
        
        all_file_specs.append(
            zds.ImagesDatasetSpecs(filenames=[group_path], data_group="image", source_axes="ZYX")
        )
        all_mask_specs.append(
            zds.MasksDatasetSpecs(filenames=[group_path], data_group="image_trabecular_mask", source_axes="ZYX")
        )


    if enable_sr_dataset:
        dataset = SRDataset(
            dataset_specs=all_file_specs+ all_mask_specs,
            patch_sampler= patch_sampler,
            return_positions=True,
            shuffle=shuffle,
            progress_bar=False,
            draw_same_chunk=draw_same_chunk,
            return_worker_id=False,
            sigma=sigma,
            downsample_factor=downsample_factor,
        )
    else:
        print("original zarr dataset")
        dataset = zds.ZarrDataset(
        dataset_specs=all_file_specs+ all_mask_specs,
        patch_sampler=patch_sampler,
        return_positions=True,
        shuffle=shuffle,
        progress_bar=False,
        draw_same_chunk=draw_same_chunk,
        return_worker_id=False,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=zds.zarrdataset_worker_init_fn,
        prefetch_factor=2,
    )
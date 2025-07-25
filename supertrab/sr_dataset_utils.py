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
import scipy.ndimage
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation, binary_erosion


def scale(image):

    scaled_image = (2.5902297 * image) + 1983.3156
    
    return scaled_image

def inverse_scale(scaled_image):
    image = (scaled_image - 1983.3156) / 2.5902297
    return image


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
    def __init__(self, sigma=1.3, downsample_factor=4, data_dim="2d", with_blur = False, override_air_values=False, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.downsample_factor = downsample_factor
        self.data_dim = data_dim
        self.with_blur = with_blur
        self.override_air_values = override_air_values
        if self.data_dim == "2d":
            self.gaussian_blur = T.GaussianBlur(kernel_size=5, sigma=sigma)
        
    
    def generate_lr(self, hr_patch):
        """
        Generate LR version of HR patch by Gaussian smoothing and downsampling.
        Supports both 2D and 3D patches.
        """
        
        if self.data_dim == "2d":
            hr_patch = hr_patch.unsqueeze(0).unsqueeze(0) 
            smoothed = self.gaussian_blur(hr_patch)
            downsampled = F.interpolate(
                smoothed,
                scale_factor=1/self.downsample_factor,
                mode="bilinear",
                align_corners=False,
            )
            restored = F.interpolate(
                downsampled,
                size=hr_patch.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            if self.with_blur:
                restored_np = gaussian_filter(restored.squeeze().cpu().numpy(), sigma=9)
                restored = torch.from_numpy(restored_np).unsqueeze(0).unsqueeze(0).to(dtype=hr_patch.dtype, device=hr_patch.device)

            
            return restored.squeeze(0).squeeze(0)


        elif self.data_dim == "3d":
            hr_np = hr_patch.numpy()
            smoothed = torch.from_numpy(
                scipy.ndimage.gaussian_filter(hr_np, sigma=self.sigma)
            )
            downsampled = F.interpolate(
                smoothed.unsqueeze(0).unsqueeze(0),  
                scale_factor=1/self.downsample_factor,
                mode="trilinear",
                align_corners=False,
            )
            restored = F.interpolate(
                downsampled,
                size=hr_patch.shape,
                mode="trilinear",
                align_corners=False,
            )

            # ADD blur for match QCT data - remove for other
            if self.with_blur:
                restored_np = gaussian_filter(restored.squeeze().cpu().numpy(), sigma=9)
                restored = torch.from_numpy(restored_np).unsqueeze(0).unsqueeze(0).to(dtype=hr_patch.dtype, device=hr_patch.device)

            return restored.squeeze(0).squeeze(0) 
        
        else:
            raise ValueError(f"Unsupported data_dim: {self.data_dim}")

    def __iter__(self):
        try:
            self._initialize()
        except Exception as e:
            print(f"EXCEPTION in __iter__: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise  
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

            if self.data_dim == "2d":
                hr_patch = torch.from_numpy(patches[0]).float()
            elif self.data_dim == "3d":
                hr_patch = torch.from_numpy(np.stack(patches)).float()

            # print(f"[DEBUG] Patch shape returned from __getitem__: {patches[0].shape}")
            # print(f"[DEBUG] patch_position: {patch_position}")


            # Safety check for dimensionality
            if self.data_dim == "2d":
                assert hr_patch.ndim == 2, f"Expected 2D patch but got shape {hr_patch.shape}"
            elif self.data_dim == "3d":
                assert hr_patch.ndim == 3, f"Expected 3D patch but got shape {hr_patch.shape}"
            else:
                raise ValueError(f"Unknown data_dim: {self.data_dim}")
            
            if self.override_air_values:
                scaled = scale(hr_patch)

                air_threshold = 1000
                marrow_mean = 2138.55
                marrow_std = 978.53

                air_mask_np = (scaled < air_threshold).cpu().numpy()
                dilated_mask_np = binary_dilation(air_mask_np, iterations=15)
                air_mask = torch.from_numpy(dilated_mask_np).to(device=scaled.device)

                num_air_voxels = air_mask.sum()

                if num_air_voxels > 0:
                    # 2. Generate marrow samples
                    marrow_samples = torch.normal(
                        mean=marrow_mean,
                        std=marrow_std,
                        size=(num_air_voxels,),
                        device=scaled.device,
                    )
                    marrow_only = torch.zeros_like(scaled)
                    marrow_only[air_mask] = marrow_samples

                    # 3. Apply Gaussian blur
                    blurred_marrow = torch.from_numpy(
                        gaussian_filter(marrow_only.cpu().numpy(), sigma=9.0)
                    ).to(device=scaled.device, dtype=scaled.dtype)

                    # 4. Erode the air mask to avoid edge artifacts
                    D, H, W = dilated_mask_np.shape
                    margin = 10  # match number of erosion iterations

                    # Create valid region (inner core where erosion is allowed)
                    valid_region = np.zeros_like(dilated_mask_np, dtype=bool)
                    valid_region[margin:D - margin, margin:H - margin, margin:W - margin] = True

                    # Perform erosion
                    eroded_temp = binary_erosion(dilated_mask_np, iterations=margin)

                    # Keep only eroded region that is inside valid core
                    eroded_mask_np = np.logical_and(eroded_temp, valid_region)

                    # Convert to PyTorch
                    insert_mask = torch.from_numpy(eroded_mask_np).to(device=scaled.device)

                    # 5. Insert blurred marrow into eroded region only
                    scaled = torch.where(insert_mask, blurred_marrow, scaled)

                hr_patch = inverse_scale(scaled)

            # Normalize HR patch to (-1, 1]
            hr_patch = hr_patch / 32768.0

            
            
            #print(f"After norm: {hr_patch.dtype}, shape: {hr_patch.shape}, min: {hr_patch.min()}, max: {hr_patch.max()}")


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


def create_dataloader(zarr_path, 
                      patch_size=(1, 128, 128), 
                      batch_size=4, 
                      num_workers=1, 
                      min_area=0.999, 
                      sigma=1.3, 
                      downsample_factor=4, 
                      draw_same_chunk = False, 
                      shuffle = True, 
                      enable_sr_dataset = True, 
                      groups_to_use=None, 
                      prefetch=2, 
                      image_group="image", 
                      mask_group="image_trabecular_mask", 
                      mask_base_path=None, 
                      data_dim="2d",
                      with_blur = False,
                      override_air_values=False):
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
    named_groups = [
        (name, root[name])
        for name in list(root.array_keys()) + list(root.group_keys())
        if groups_to_use is None or name in groups_to_use
    ]
    if groups_to_use is not None:
        named_groups = [(name, group) for name, group in named_groups if name in groups_to_use]
    print(f"Found {len(named_groups)} groups:")
    for name, _ in named_groups:
        print(f"  - {name}")

    #print(f"Patch size to sampler: {patch_size}")

    patch_sampler = zds.PatchSampler(patch_size, min_area=min_area)

    all_file_specs = []
    all_mask_specs = []

    for name, _ in named_groups:
        all_file_specs.append(
            zds.ImagesDatasetSpecs(
                filenames=[str(zarr_path)],
                data_group=f"{name}/{image_group}", 
                source_axes="ZYX"
            )
        )

        # print(f"[DATA] Using image group: {name}/{image_group}")


        if mask_base_path is not None:
            mask_path = f"{name}/{mask_base_path}/{mask_group}"
        else:
            mask_path = f"{name}/{mask_group}" 

        # print(f"[MASK] Using mask group: {mask_path}")

        all_mask_specs.append(
            zds.MasksDatasetSpecs(
                filenames=[str(zarr_path)],
                data_group=mask_path,
                source_axes="ZYX"
            )
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
            data_dim=data_dim,
            with_blur=with_blur,
            override_air_values=override_air_values,
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
        prefetch_factor=prefetch,
    )


# mixed dataset =======================================================================

class TripletZarrDataset(torch.utils.data.Dataset):
    def __init__(self, zarr_path, group_names, patch_size=(1, 256, 256), conditioning_mode="qct"):
        self.conditioning_mode = conditioning_mode
        self.patches = []

        for group_name in group_names:
            group = zarr.open(str(zarr_path), mode="r")[group_name]
            qct = group["qct"]
            hrpqct = group["hrpqct"]
            lr = group["lr"]

            num_patches = qct.shape[0]
            for i in range(num_patches):
                self.patches.append((group_name, i))

        self.z = zarr.open(str(zarr_path), mode="r")
        self.patch_size = patch_size

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        group_name, idx = self.patches[index]
        group = self.z[group_name]

        hr_image = torch.tensor(group["hrpqct"][idx], dtype=torch.float32)
        qct = torch.tensor(group["qct"][idx], dtype=torch.float32)
        lr = torch.tensor(group["lr"][idx], dtype=torch.float32)

        if self.conditioning_mode == "qct":
            conditioning = qct
        elif self.conditioning_mode == "lr":
            conditioning = lr
        elif self.conditioning_mode == "mix":
            conditioning = qct if torch.rand(1).item() < 0.5 else lr
        else:
            raise ValueError(f"Invalid conditioning_mode: {self.conditioning_mode}")

        return {
            "hr_image": hr_image,
            "conditioning": conditioning,
            "qct": qct,
            "lr": lr,
            "group": group_name,
            "index": idx,
        }

def create_triplet_dataloader(zarr_path, group_names, conditioning_mode="qct", patch_size=(1, 256, 256), batch_size=4, num_workers=0):
    dataset = TripletZarrDataset(zarr_path, group_names, patch_size, conditioning_mode)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

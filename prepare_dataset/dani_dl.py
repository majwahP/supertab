
import torchvision.transforms.functional as TF
import zarrdataset as zds
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import zarr
from collections import Counter
from functools import partial
import PIL
from PIL import Image
import cv2
import albumentations

class ImageSample():
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
        del self._ordering
        self._ordering = None

    def next_patch(self):
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
    
class LungSR(zds.ZarrDataset):
    def __init__(self, size=None,
                 degradation=None, downscale_f=4, min_crop_f=0.5, max_crop_f=1.,
                 random_crop=True, **kwargs):
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        print(f"{self.LR_size=}")
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

        self.pil_interpolation = False # gets reset later if incase interp_op is from pillow

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)

        else:
            interpolation_fn = {
            "cv_nearest": cv2.INTER_NEAREST,
            "cv_bilinear": cv2.INTER_LINEAR,
            "cv_bicubic": cv2.INTER_CUBIC,
            "cv_area": cv2.INTER_AREA,
            "cv_lanczos": cv2.INTER_LANCZOS4,
            "pil_nearest": PIL.Image.NEAREST,
            "pil_bilinear": PIL.Image.BILINEAR,
            "pil_bicubic": PIL.Image.BICUBIC,
            "pil_box": PIL.Image.BOX,
            "pil_hamming": PIL.Image.HAMMING,
            "pil_lanczos": PIL.Image.LANCZOS,
            }[degradation]

            self.pil_interpolation = degradation.startswith("pil_")

            if self.pil_interpolation:
                self.degradation_process = partial(TF.resize, size=self.LR_size, interpolation=interpolation_fn)

            else:
                self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
                                                                          interpolation=interpolation_fn)

        super().__init__(**kwargs)

    def __iter__(self):
        # Preload the files and masks associated with them
        self._initialize()
    
        samples = [
            ImageSample(im_id, chk_id, shuffle=self._shuffle)
            for im_id in range(len(self._arr_lists))
            for chk_id in range(len(self._toplefts[im_id]))
        ]

        # Shuffle chunks here if samples will come from the same chunk until
        # they are depleted.
        if self._shuffle and self._draw_same_chunk:
            random.shuffle(samples)

        prev_im_id = -1
        prev_chk_id = -1
        prev_chk = -1
        curr_chk = 0
        self._curr_collection = None

        while samples:
            # Shuffle chunks here if samples can come from different chunks.
            if self._shuffle and not self._draw_same_chunk:
                curr_chk = random.randrange(0, len(samples))

            im_id = samples[curr_chk].im_id
            chk_id = samples[curr_chk].chk_id

            chunk_tlbr = self._toplefts[im_id][chk_id]

            # If this sample is from a different image or chunk, free the
            # previous sample and re-sample the patches from the current chunk.
            if prev_im_id != im_id or chk_id != prev_chk_id:
                if prev_chk >= 0:
                    # Free the patch ordering from the previous chunk to save
                    # memory.
                    samples[prev_chk].free_sampler()

                prev_chk = curr_chk
                prev_chk_id = chk_id

                if prev_im_id != im_id:
                    prev_im_id = im_id
                    self._curr_collection = self._arr_lists[im_id]

                if self._patch_sampler is not None:
                    patches_tls = self._patch_sampler.compute_patches(
                        self._curr_collection,
                        chunk_tlbr
                    )

                else:
                    patches_tls = [chunk_tlbr]

                samples[curr_chk].num_patches = len(patches_tls)

                if not len(patches_tls):
                    samples.pop(curr_chk)
                    prev_chk = -1
                    continue

            # # Initialize the count of top-left positions for patches inside
            # # this chunk.
            # if samples[curr_chk].num_patches is None:

            curr_patch, is_empty = samples[curr_chk].next_patch()

            # When all possible patches have been extracted from the current
            # chunk, remove that chunk from the list of samples.
            if is_empty:
                samples.pop(curr_chk)
                prev_chk = -1

            patch_tlbr = patches_tls[curr_patch]
            patches = self.__getitem__(patch_tlbr)[0]
            
            # print(len(patches))
            # print(patches.shape)

            if patches.shape[0] > 1:
                raise ValueError("Only single channel images are supported")
            sample = patches[0]

            LR_image = self.degradation_process(Image.fromarray(sample))

            HR_image = sample #uint16_norm(sample)
            LR_image = (np.array(LR_image))
            #LR_image = uint16_norm(np.array(LR_image))

            example = {}
            example["image"] = (HR_image).astype(np.float32)
            example["LR_image"] = (LR_image).astype(np.float32)
            


            if self._return_positions:
                pos = [
                    [patch_tlbr[ax].start
                     if patch_tlbr[ax].start is not None else 0,
                     patch_tlbr[ax].stop
                     if patch_tlbr[ax].stop is not None else -1
                    ] if ax in patch_tlbr else [0, -1]
                    for ax in self._collections[self._ref_mod][0]["axes"]
                ]
                # patches = [np.array(pos, dtype=np.int64)] + patches
                example["position"] = np.array(pos, dtype=np.int64)

            if self._return_worker_id:
                wid = [np.array(self._worker_id, dtype=np.int64)]
                # patches = wid + patches
                example["wid"] = wid

            # if len(patches) > 1:
            #     patches = tuple(patches)
            # else:
            #     patches = patches[0]
            # print(f'{example["image"].shape=}')
            # print(f'{example["LR_image"].shape=}')
            yield example



class LungSRTrain(LungSR):
    def __init__(self, **kwargs):
        #DATASRC="/das/work/units/tomcat/p21219/Data20/data/pneumotomo"
        #path = Path(DATASRC, "m_G_double", "scans.zarr")
        zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab_testf32_128x512x512.zarr")
        root: zarr.hierarchy.Group = zarr.open(str(zarr_path))
        # self.num_records = 78304
        size = kwargs["size"]
        patch_size = (1, size, size)
        zarr_patch_sampler = zds.PatchSampler(patch_size, min_area=0.25)

        groups = [g[1] for g in root.groups()][:1]
        print('Train groups:')
        print(f"{groups=}")
        print(type(groups[0]))
        file_specs = zds.ImagesDatasetSpecs(
            filenames=groups,
            data_group="image",
            source_axes="ZYX",
        )
        # mask_2d = np.array([
        #     [0,0,0,0],
        #     [0,0,1,0],
        #     [0,1,0,0],
        #     [0,0,0,0],
        # ], dtype=bool)
        # mask = np.stack([mask_2d, mask_2d, mask_2d])
        mask = zarr.load(zarr_path, path=f"1955_L/image_trabecular_mask")
        #mask = da.from_zarr(zarr_path, component=f"1955_L/image_trabecular_mask").compute()
        # Dummy mask for one chunk
        print(f"Mask shape: {mask.shape}")
        print(f"Mask dtype: {mask.dtype}")
        print(f"Mask sum: {mask.sum()} (total number of True pixels)")
        masks_specs = zds.MasksDatasetSpecs(
            filenames=mask,
            source_axes="ZYX",
        )
        # masks_specs = zds.MasksDatasetSpecs(
        #     filenames=groups,
        #     data_group="image_trabecular_mask",
        #     source_axes="ZYX",
        # )
        # named_groups = list(root.groups())
        # print(f"Found {len(named_groups)} groups:")
        # for name, _ in named_groups:
        #     print(f"  - {name}")
        # all_file_specs = []
        # all_mask_specs = []
        # all_groups = [f"{zarr_path}/{name}" for name, _ in named_groups]

        # #debug, try with first group
        # name, group = named_groups[0]
        # first_group_path = f"{zarr_path}/{name}"

        # all_file_specs.append(
        #     zds.ImagesDatasetSpecs(filenames=[first_group_path], data_group="image", source_axes="ZYX")
        # )
        # file_specs = zds.ImagesDatasetSpecs(filenames=[first_group_path], data_group="image", source_axes="ZYX")
        # all_mask_specs.append(
        #     zds.MasksDatasetSpecs(filenames=[first_group_path], data_group="image_trabecular_mask", source_axes="ZYX")
        # )
        # masks_specs = zds.MasksDatasetSpecs(filenames=[first_group_path], data_group="image_trabecular_mask", source_axes="ZYX")
        super().__init__(dataset_specs=[file_specs, masks_specs],
        patch_sampler=zarr_patch_sampler,
        return_positions=True,
        shuffle=True,
        progress_bar=True,
        draw_same_chunk=False,
        return_worker_id=False,**kwargs)

class LungSRValidation(LungSR):
    def __init__(self, **kwargs):
        DATASRC="/das/work/units/tomcat/p21219/Data20/data/pneumotomo"
        path = Path(DATASRC, "m_G_double", "scans.zarr")
        root: zarr.hierarchy.Group = zarr.open(str(path))
        # self.num_records = 78304
        size = kwargs["size"]
        patch_size = (1, size, size)
        zarr_patch_sampler = zds.PatchSampler(patch_size, min_area=0.25)

        # groups = [g[1] for g in self.root.groups()][0]
        groups = [g[1] for g in root.groups()][-2:]
        print('Train groups:')
        print(f"{groups=}")
        # groups = [g[1] for g in self.root.groups()][:8]
        file_specs = zds.ImagesDatasetSpecs(
            filenames=groups,
            data_group="im1",
            source_axes="ZYX",
        )
        masks_specs = zds.MasksDatasetSpecs(
            filenames=groups,
            data_group="im1_mask",
            source_axes="ZYX",
        )
        super().__init__(dataset_specs=[file_specs, masks_specs],
        patch_sampler=zarr_patch_sampler,
        return_positions=True,
        shuffle=True,
        progress_bar=True,
        draw_same_chunk=False,
        return_worker_id=False,**kwargs)

class LungSRTest(LungSR):
    def __init__(self, **kwargs):
        DATASRC="/das/work/units/tomcat/p21219/Data20/data/pneumotomo"
        path = Path(DATASRC, "m_G_double", "scans.zarr")
        root: zarr.hierarchy.Group = zarr.open(str(path))
        # self.num_records = 78304
        size = kwargs["size"]
        patch_size = (1, size, size)
        zarr_patch_sampler = zds.PatchSampler(patch_size, min_area=0.25)

        # groups = [g[1] for g in self.root.groups()][0]
        groups = [g[1] for g in root.groups()][5:8]
        print('Train groups:')
        print(f"{groups=}")
        # groups = [g[1] for g in self.root.groups()][:8]
        file_specs = zds.ImagesDatasetSpecs(
            filenames=groups,
            data_group="im1",
            source_axes="ZYX",
        )
        masks_specs = zds.MasksDatasetSpecs(
            filenames=groups,
            data_group="im1_mask",
            source_axes="ZYX",
        )
        super().__init__(dataset_specs=[file_specs, masks_specs],
        patch_sampler=zarr_patch_sampler,
        return_positions=True,
        shuffle=True,
        progress_bar=True,
        draw_same_chunk=False,
        return_worker_id=False,**kwargs)
        
        
        
if __name__ == "__main__":
    
    dataset = LungSRTrain(
        size=64, degradation="pil_bilinear"
    )
    print("dataset")
    # dataset = LungSRTrain(size=512, degradation="pil_nearest")
    dl = DataLoader(dataset, batch_size=4,
                          num_workers=2 ,
                        #   worker_init_fn=worker_init_fn)
                          worker_init_fn=zds.zarrdataset_worker_init_fn)
    print("dataloader")

    samples = 0
    positions = Counter()
    # for batch in tqdm(iter(dataset)):
    print("strating loop")
    for batch in tqdm(dl):
        print("in loop")
        # breakpoint()
        samples += batch["image"].shape[0]
        print(f"{samples=}")
        #positions.update(tensor_to_string(batch["position"]))
        # mc = positions.most_common(1)[0]
        # if mc[1] > 2:
        #     print(f"Found position: {mc[0]}. more than once")
        #     break
        # print(positions)
        if samples > 10:
            break

    print("Total seen samples:")
    print(samples)

    print("Top 10 positions in the counter:")
    print(positions.most_common(10))

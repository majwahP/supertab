import dask.array as da
from dask.array import coarsen
import zarr
from pathlib import Path

zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab_testf32.zarr")

src = da.from_zarr(zarr_path / "1955_L" / "image_trabecular_mask")#.astype("float32")
save_path = zarr_path.parent / "supertrab_testf32_128x512x512.zarr" / "1955_L" 

num_slices = src.shape
# half = num_slices // 2
print(num_slices)
# print(half)

# top_half = src[:half]
# bottom_half = src[half:]

# print("saving top part")
# top_half.to_zarr(save_path / "image_trabecular_mask_top", overwrite=True)
# print("saving bottom part")
# bottom_half.to_zarr(save_path / "image_trabecular_mask_bottom", overwrite=True)


src.to_zarr(save_path / "image_trabecular_mask", overwrite=True)


#binning mask
# summed = coarsen(src.sum, src, {0: 4, 1: 4, 2: 4})
# binned_bool = summed > 0
# save_path = zarr_path.parent / "supertrab_testf32.zarr" / "1955_L"
# print(binned_bool.shape)
# binned_bool.to_zarr(save_path / "image_trabecular_mask_bottom_binned", overwrite=True)

print("Done")
import dask.array as da
from dask.diagnostics import ProgressBar
import zarr


group = "1955_L"
root = zarr.open("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr", mode="r")
image_array = da.from_zarr(root[group]["image"])

with ProgressBar():
    global_min = image_array.min().compute()
    global_max = image_array.max().compute()

print(f"Global min: {global_min}")
print(f"Global max: {global_max}")

import dask.array as da
import zarr
from pathlib import Path

zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
image_path = zarr_path / "2019_L" / "image"

image = da.from_zarr(image_path)

zip_path = zarr_path / "2019_L" / "image.zip"
store = zarr.ZipStore(str(zip_path), mode='w') 

image.to_zarr(store, component=None, overwrite=True)

store.close()

print(f"Saved image to {zip_path} as a single .zip Zarr store.")

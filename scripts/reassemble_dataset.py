import dask.array as da
from pathlib import Path



# Paths
zarr_path=Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
base_path = zarr_path / "2019_L" / "sr_volume_256_QCT_ds10_blur_model_with_scaling" 
part1_path = base_path / "part_1"
split_part2_root = base_path / "part_2_split"
output_path = base_path / "reassembled"

# Load part_1
print("Loading part_1...")
part1 = da.from_zarr(part1_path)

# Load the 16 sub-parts of part_2_split
print("Loading part_2_split...")
part2_chunks = []
for i in range(1, 17):
    subpart_path = split_part2_root / f"part_{i}"
    print(f"  - {subpart_path}")
    part = da.from_zarr(subpart_path)
    part2_chunks.append(part)

# Concatenate part_2
part2 = da.concatenate(part2_chunks, axis=0)

# Combine part_1 and part_2
full_volume = da.concatenate([part1, part2], axis=0)

# Save to a new Zarr store
print(f"Saving reassembled volume to: {output_path}")
full_volume.to_zarr(output_path, overwrite=True)

print("Done reassembling.")

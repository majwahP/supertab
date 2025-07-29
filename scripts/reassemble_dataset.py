import dask.array as da
from pathlib import Path



# Paths
zarr_path=Path("/usr/terminus/data-xrm-01/stamplab/RESTORE/supertrab.zarr")
# base_path = zarr_path / "1955_L" / "image_split" 
# part1_path = base_path / "part_1"
# split_part2_root = base_path / "part_2_split"
# output_path = base_path / "reassembled"

base_path = zarr_path / "2019_L" / "sr_volume_256_10_200ep_given_QCT" 
part1_path = base_path / "part_1"
split_part2_root = base_path / "part_2_split"
output_path = base_path / "reassembled_3"

# Load part_1
print("Loading part_1...")
part1 = da.from_zarr(part1_path)

print(f"  - Shape: {part1.shape}")
print(f"  - Dtype: {part1.dtype}")
print(f"  - Min: {part1.min().compute()}, Max: {part1.max().compute()}")

# Load the 16 sub-parts of part_2_split
print("Loading part_2_split...")
part2_chunks = []
for i in range(1, 17):
    subpart_path = split_part2_root / f"part_{i}"
    print(f"  - {subpart_path}")
    part = da.from_zarr(subpart_path)
    print(f"    > Shape: {part.shape}, Min: {part.min().compute()}, Max: {part.max().compute()}")

    part2_chunks.append(part)

# Concatenate part_2
print("\n🔹 Concatenating part_2...")
part2 = da.concatenate(part2_chunks, axis=0)
print(f"  - part2 shape: {part2.shape}")
print(f"  - part2 min/max: {part2.min().compute()} / {part2.max().compute()}")

# Combine part_1 and part_2
print("\n🔹 Combining part_1 and part_2...")
full_volume = da.concatenate([part1, part2], axis=0)
print(f"  - full_volume shape: {full_volume.shape}")
print(f"  - full_volume min/max: {full_volume.min().compute()} / {full_volume.max().compute()}")

# Save to a new Zarr store
print(f"Saving reassembled volume to: {output_path}")
full_volume.to_zarr(output_path, overwrite=False)

print("Done reassembling.")

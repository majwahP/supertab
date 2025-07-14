"""
Script for inspecting and validating patch sampling from a Zarr-based super-resolution dataset.

This script:
- Initializes a dataloader for HR/LR patch pairs from a Zarr dataset.
- Plots a random selection of HR/LR patch pairs and saves the result as an image grid.
- Checks for duplicate patch positions and verifies whether duplicated patches are identical.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from supertrab.sr_dataset_utils import create_dataloader
from supertrab.analysis_utils import plot_random_samples_from_dataloader, check_patch_uniqueness


def main():
    zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/supertrab.zarr")
    output_path = "samples/images/random_patches.png"

    groups = ["2019_L", "1955_L", "1956_L", "1996_R", "2005_L", "2007_L"]

    dataloader = create_dataloader(zarr_path, groups_to_use=groups, patch_size=(1,256,256), downsample_factor=10)
    print("done creating dataloader")

    print("Plotting samples")
    plot_random_samples_from_dataloader(dataloader, output_path, max_samples=21)

    # print("Check uniqueness")
    # check_patch_uniqueness(dataloader)

    print("Done!")


if __name__ == "__main__":
    main()

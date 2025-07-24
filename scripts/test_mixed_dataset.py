import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from supertrab.sr_dataset_utils import create_triplet_dataloader

def normalize(img):
    img = img - img.min()
    img = img / img.max()
    return img

# --- CONFIG ---
zarr_path = Path("/usr/terminus/data-xrm-01/stamplab/external/tacosound/HR-pQCT_II/zarr_data/paired_patch_dataset.zarr")
group_names = ["1955_L"]  # You can include more groups if desired
num_samples = 10
batch_size = 1  # Load one patch at a time

# --- Dataset and Dataloader ---
# dataloader = create_triplet_dataloader(
#     zarr_path=zarr_path,
#     group_names=group_names,
#     conditioning_mode="qct",  
#     patch_size=(1, 256, 256),
#     batch_size=batch_size,
#     num_workers=0
# )

dataloader = create_triplet_dataloader(
    zarr_path=zarr_path,
    group_names=group_names,
    conditioning_mode="mixed",  # <- TESTING MIXED MODE
    patch_size=(1, 256, 256),
    batch_size=batch_size,
    num_workers=0
)

# --- Collect 10 samples ---
# qct_list, lr_list, hrpqct_list = [], [], []

# for i, sample in enumerate(dataloader):
#     if i >= num_samples:
#         break
#     qct_list.append(sample["qct"][0])
#     lr_list.append(sample["lr"][0])
#     hrpqct_list.append(sample["hr_image"][0])

conditioning_list, hr_list = [], []

for i, sample in enumerate(dataloader):
    if i >= num_samples:
        break
    conditioning_list.append(sample["conditioning"][0])
    hr_list.append(sample["hr_image"][0])

# --- Stack into tensors ---
conditioning = torch.stack(conditioning_list)
hr = torch.stack(hr_list)

# # --- Stack into tensors for plotting ---
# qct = torch.stack(qct_list)
# lr = torch.stack(lr_list)
# hrpqct = torch.stack(hrpqct_list)

# --- Plot Grid ---
# cols = 3
# rows = num_samples
# fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

# for i in range(num_samples):
#     axs[i, 0].imshow(normalize(qct[i, 0]), cmap="gray")
#     axs[i, 0].set_title(f"QCT {i}")
#     axs[i, 1].imshow(normalize(lr[i, 0]), cmap="gray")
#     axs[i, 1].set_title(f"LR {i}")
#     axs[i, 2].imshow(normalize(hrpqct[i, 0]), cmap="gray")
#     axs[i, 2].set_title(f"HR-pQCT {i}")
#     for j in range(3):
#         axs[i, j].axis("off")

# plt.tight_layout()
# Path("patch_outputs").mkdir(parents=True, exist_ok=True)
# plt.savefig("patch_outputs/10_random_triplets_from_dataloader.png", dpi=300)

cols = 2
rows = num_samples
fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

for i in range(num_samples):
    axs[i, 0].imshow(normalize(conditioning[i, 0]), cmap="gray")
    axs[i, 0].set_title(f"Conditioning {i}")
    axs[i, 1].imshow(normalize(hr[i, 0]), cmap="gray")
    axs[i, 1].set_title(f"HR-pQCT {i}")
    for j in range(2):
        axs[i, j].axis("off")

plt.tight_layout()
Path("patch_outputs").mkdir(parents=True, exist_ok=True)
plt.savefig("patch_outputs/10_random_mixed_conditioning_samples.png", dpi=300)
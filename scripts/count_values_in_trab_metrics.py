import pandas as pd
from pathlib import Path

# --- CONFIG ---
output_dir = Path("stat_outputs")
ds_factor = 10  # Set the DS factor you want to inspect
csv_file = output_dir / f"all_patch_metrics_ds{ds_factor}_otsu.csv"

# --- Load CSV ---
df = pd.read_csv(csv_file)

# --- Count unique patches (by index or position) ---
num_unique_patches = df["patch_idx"].nunique()
positions = df["position"].nunique() if "position" in df.columns else "N/A"

print(f"Downsampling factor: {ds_factor}")
print(f"Total unique patch indices in metrics CSV: {num_unique_patches}")
print(f"Total unique patch positions (if present): {positions}")

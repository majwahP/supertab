import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Downsampling factors to include
DS_FACTORS = [4, 6, 8]#, 10]

# Path to your CSVs
project_root = Path(__file__).resolve().parents[1]
stat_dir = project_root / "stat_outputs"

# Load and combine all datasets
df_all = []
for ds in DS_FACTORS:
    csv_path = stat_dir / f"all_patch_metrics_ds{ds}.csv"
    df_tmp = pd.read_csv(csv_path)
    df_tmp["ds_factor"] = ds
    df_all.append(df_tmp)

# Combine into one dataframe
df = pd.concat(df_all, ignore_index=True)

# Output folders
output_dir_all = "boxplots_across_ds"
output_dir_hr_sr = "boxplots_across_ds_hr_sr_only"
os.makedirs(output_dir_all, exist_ok=True)
os.makedirs(output_dir_hr_sr, exist_ok=True)

# Metric columns
exclude_cols = {"position", "patch_idx", "source", "ds_factor"}
metric_cols = [col for col in df.columns if col not in exclude_cols]

# Loop through each metric
for metric in metric_cols:
    # --- Plot 1: All sources (HR, LR, SR) ---
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.boxplot(data=df, x="ds_factor", y=metric, hue="source", palette="pastel")
    sns.stripplot(data=df, x="ds_factor", y=metric, hue="source", 
                  dodge=True, color="black", alpha=0.3, jitter=0.2, size=2)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend_.remove()

    plt.legend(handles[:3], labels[:3], title="Source")
    plt.title(metric.replace("_", " ").title())
    plt.xlabel("Downsampling Factor")
    plt.ylabel(metric)
    plt.tight_layout()
    save_path = os.path.join(output_dir_all, f"{metric}_HR_LR_SR_by_ds.png")
    plt.savefig(save_path)
    plt.close()

    # --- Plot 2: HR vs SR only ---
    df_hr_sr = df[df["source"].isin(["HR", "SR"])]
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.boxplot(data=df_hr_sr, x="ds_factor", y=metric, hue="source", palette="pastel")
    sns.stripplot(data=df_hr_sr, x="ds_factor", y=metric, hue="source", 
                  dodge=True, color="black", alpha=0.3, jitter=0.2, size=2)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend_.remove()

    plt.legend(handles[:2], labels[:2], title="Source")
    plt.title(metric.replace("_", " ").title() + " (HR vs SR only)")
    plt.xlabel("Downsampling Factor")
    plt.ylabel(metric)
    plt.tight_layout()
    save_path_hr_sr = os.path.join(output_dir_hr_sr, f"{metric}_HR_SR_by_ds.png")
    plt.savefig(save_path_hr_sr)
    plt.close()

print("Boxplots saved for all DS factors")

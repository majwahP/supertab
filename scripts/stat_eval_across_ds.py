import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Downsampling factors to include
DS_FACTORS = [4, 6, 8, 10]  

# Path to your CSVs
project_root = Path(__file__).resolve().parents[1]
stat_dir = project_root / "stat_outputs"

# Load and combine all datasets
df_all = []
for ds in DS_FACTORS:
    csv_path = stat_dir / f"all_patch_metrics_ds{ds}_otsu.csv"
    df_tmp = pd.read_csv(csv_path)
    df_tmp["ds_factor"] = ds
    df_all.append(df_tmp)

# Combine into one dataframe
df = pd.concat(df_all, ignore_index=True)

# Output subfolders
output_dir = "boxplots_across_ds_otsu"
box_dir = os.path.join(output_dir, "boxplots_all_sources")
box_hrsr_dir = os.path.join(output_dir, "boxplots_hr_sr_only")
violin_dir = os.path.join(output_dir, "violinplots_all_sources")
violin_hrsr_dir = os.path.join(output_dir, "violinplots_hr_sr_only")

plt.rcParams.update({
    'axes.titlesize': 18,       # Title font size
    'axes.labelsize': 16,       # X and Y label size
    'xtick.labelsize': 14,      # X tick label size
    'ytick.labelsize': 14,      # Y tick label size
    'legend.fontsize': 14,      # Legend font size
    'legend.title_fontsize': 15 # Legend title font size
})

for d in [box_dir, box_hrsr_dir, violin_dir, violin_hrsr_dir]:
    os.makedirs(d, exist_ok=True)

# Metric columns
exclude_cols = {"position", "patch_idx", "source", "ds_factor"}
metric_cols = [col for col in df.columns if col not in exclude_cols]

# Loop through each metric
for metric in metric_cols:
    # --- Boxplot: All sources (HR, LR, SR) ---
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.boxplot(data=df, x="ds_factor", y=metric, hue="source", palette="pastel")
    sns.stripplot(data=df, x="ds_factor", y=metric, hue="source", 
                  dodge=True, color="black", alpha=0.3, jitter=0.2, size=2, legend=False)
    plt.title(metric.replace("_", " ").title())
    plt.xlabel("Downsampling Factor")
    plt.ylabel(metric)
    plt.legend(title="Source", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(box_dir, f"{metric}_HR_LR_SR_by_ds.png"))
    plt.close()

    # --- Boxplot: HR vs SR only ---
    df_hr_sr = df[df["source"].isin(["HR", "SR"])]
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.boxplot(data=df_hr_sr, x="ds_factor", y=metric, hue="source", palette="pastel")
    sns.stripplot(data=df_hr_sr, x="ds_factor", y=metric, hue="source", 
                  dodge=True, color="black", alpha=0.3, jitter=0.2, size=2, legend=False)
    plt.title(metric.replace("_", " ").title() + " (HR vs SR only)")
    plt.xlabel("Downsampling Factor")
    plt.ylabel(metric)
    plt.legend(title="Source", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(box_hrsr_dir, f"{metric}_HR_SR_by_ds.png"))
    plt.close()

    # --- Violinplot: All sources ---
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.violinplot(data=df, x="ds_factor", y=metric, hue="source", palette="pastel", inner=None)
    sns.stripplot(data=df, x="ds_factor", y=metric, hue="source", 
                  dodge=True, color="black", alpha=0.3, jitter=0.2, size=2, legend=False)
    plt.title(metric.replace("_", " ").title())
    plt.xlabel("Downsampling Factor")
    plt.ylabel(metric)
    plt.legend(title="Source", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(violin_dir, f"{metric}_HR_LR_SR_violin_by_ds.png"))
    plt.close()

    # --- Violinplot: HR vs SR only ---
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.violinplot(data=df_hr_sr, x="ds_factor", y=metric, hue="source", palette="pastel", inner=None)
    sns.stripplot(data=df_hr_sr, x="ds_factor", y=metric, hue="source", 
                  dodge=True, color="black", alpha=0.3, jitter=0.2, size=2, legend=False)
    plt.title(metric.replace("_", " ").title() + " (HR vs SR Violin)")
    plt.xlabel("Downsampling Factor")
    plt.ylabel(metric)
    plt.legend(title="Source", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(violin_hrsr_dir, f"{metric}_HR_SR_violin_by_ds.png"))
    plt.close()

print(f"All plots saved in '{output_dir}'.")

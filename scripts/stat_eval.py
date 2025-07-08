import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

DS_FACTOR = 8

# Path to your CSV
project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / "stat_outputs" / f"all_patch_metrics_ds{DS_FACTOR}.csv"

# Load the data
df = pd.read_csv(csv_path)

# Output folder for plots
output_dir = "boxplots"
os.makedirs(output_dir, exist_ok=True)

# List of metrics to plot (exclude metadata columns)
exclude_cols = {"position", "patch_idx", "source"}
metric_cols = [col for col in df.columns if col not in exclude_cols]

# Loop through each metric
for metric in metric_cols:
    # --- Plot 1: HR, LR, SR ---
    plt.figure(figsize=(6, 5), dpi=300)
    ax = sns.boxplot(data=df, x="source", y=metric, palette="pastel", width=0.6)
    sns.stripplot(data=df, x="source", y=metric, color="black", alpha=0.5, jitter=0.2, size=2)

    legend = ax.get_legend()
    if legend:
        legend.remove()

    plt.title(metric.replace("_", " ").title())
    plt.xlabel("")
    plt.ylabel(metric)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{metric}_HR_LR_SR_ds{DS_FACTOR}.png")
    plt.savefig(save_path)
    plt.close()

    # --- Plot 2: HR vs SR only ---
    df_hr_sr = df[df["source"].isin(["HR", "SR"])]

    plt.figure(figsize=(6, 5), dpi=300)
    ax = sns.boxplot(data=df_hr_sr, x="source", y=metric, palette="pastel", width=0.6)
    sns.stripplot(data=df_hr_sr, x="source", y=metric, color="black", alpha=0.5, jitter=0.2, size=2)

    legend = ax.get_legend()
    if legend:
        legend.remove()

    plt.title(metric.replace("_", " ").title() + " (HR vs SR)")
    plt.xlabel("")
    plt.ylabel(metric)
    plt.tight_layout()
    save_path_hr_sr = os.path.join(output_dir, f"{metric}_HR_SR_ds{DS_FACTOR}.png")
    plt.savefig(save_path_hr_sr)
    plt.close()

print(f"All boxplots saved to '{output_dir}'")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Settings ---
INPUT_CSV = "metric_outputs/trabecular_metrics_LR_HR_HRblurred.csv"
OUTPUT_DIR = "metric_outputs/box_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load data ---
df = pd.read_csv(INPUT_CSV)

# --- Correct metric names based on actual CSV ---
metrics = [
    "bone_volume_fraction",
    "trabecular_thickness_mean",
    "trabecular_spacing_mean",
    "trabecular_number"
]

# --- Plot boxplots ---
for metric in metrics:
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x="source", y=metric, palette="Set2")
    plt.title(f"Distribution of {metric.replace('_', ' ').capitalize()}")
    plt.ylabel(metric.replace("_", " ").capitalize())
    plt.xlabel("Source")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"{metric}_boxplot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

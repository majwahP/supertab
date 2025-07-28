import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import shapiro, wilcoxon
import numpy as np

# Downsampling factors to include
DS_FACTORS = [4, 6, 8, 10]

def get_ylabel(metric):
    return metric + " (%)" if metric == "bone_volume_fraction" else metric

# Path to your CSVs
project_root = Path(__file__).resolve().parents[1]
stat_dir = project_root / "stat_outputs"

# Load and combine all datasets
df_all = []
for ds in DS_FACTORS:
    if ds == 4:
        csv_path = stat_dir / f"trabecular_metrics_ds{4}_200ep.csv"
    else: 
        csv_path = stat_dir / f"all_patch_metrics_ds{ds}_otsu.csv"
    df_tmp = pd.read_csv(csv_path)
    df_tmp["ds_factor"] = ds
    df_all.append(df_tmp)

# Combine into one dataframe
df = pd.concat(df_all, ignore_index=True)

if "bone_volume_fraction" in df.columns:
    df["bone_volume_fraction"] = df["bone_volume_fraction"] * 100

exclude_cols = {"position", "patch_idx", "source", "ds_factor"}
metric_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.float32, np.int64]]


# Metric columns
exclude_cols = {"position", "patch_idx", "source", "ds_factor"}
metric_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.float32, np.int64]]


# Output subfolders
output_dir = "boxplots_across_ds_otsu"
box_dir = os.path.join(output_dir, "boxplots_all_sources")
box_hrsr_dir = os.path.join(output_dir, "boxplots_hr_sr_only")
violin_dir = os.path.join(output_dir, "violinplots_all_sources")
violin_hrsr_dir = os.path.join(output_dir, "violinplots_hr_sr_only")

plt.rcParams.update({
    'axes.titlesize': 26,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'legend.title_fontsize': 16
})

for d in [box_dir, box_hrsr_dir, violin_dir, violin_hrsr_dir]:
    os.makedirs(d, exist_ok=True)

# Loop through each metric
for metric in metric_cols:
    # ==================== All Sources (HR, LR, SR) - BOXPLOT ====================
    df_all_sources = df.copy()
    for draw_lines in [False, True]:
        plt.figure(figsize=(10, 6), dpi=300)
        ax = sns.boxplot(data=df_all_sources, x="ds_factor", y=metric, hue="source", palette="pastel")
        sns.stripplot(data=df_all_sources, x="ds_factor", y=metric, hue="source",
                      dodge=True, color="black", alpha=0.3, jitter=0.2, size=2, legend=False)

        if draw_lines:
            hue_order = ["HR", "LR", "SR"]
            hue_offsets = {"HR": -0.25, "LR": 0.0, "SR": 0.25}
            positions_by_ds_source = {
                ds: {source: i + hue_offsets[source] for source in hue_order}
                for i, ds in enumerate(sorted(df_all_sources["ds_factor"].unique()))
            }

            df_pivot_all = df_all_sources.pivot_table(index=["ds_factor", "patch_idx"], columns="source", values=metric)
            for (ds, patch_idx), row in df_pivot_all.dropna().iterrows():
                sources = [s for s in hue_order if s in row.index]
                xs = [positions_by_ds_source[ds][s] for s in sources]
                ys = [row[s] for s in sources]
                if len(xs) >= 2:
                    ax.plot(xs, ys, color="gray", alpha=0.3, linewidth=0.8)
                    ax.scatter(xs, ys, color="black", s=10, alpha=0.5)

        plt.title(metric.replace("_", " ").title())
        plt.xlabel("Downsampling Factor")
        ylabel = get_ylabel(metric)
        plt.ylabel(ylabel)
        plt.legend(title="Source", loc="upper right")
        plt.tight_layout()
        fname = f"{metric}_HR_LR_SR_by_ds"
        if draw_lines:
            fname += "_lines"
        plt.savefig(os.path.join(box_dir, f"{fname}.png"))
        plt.close()

    # ==================== HR vs SR Only - BOXPLOT ====================
    df_hr_sr = df[df["source"].isin(["HR", "SR"])]
    for draw_lines in [False, True]:
        plt.figure(figsize=(10, 6), dpi=300)
        ax = sns.boxplot(data=df_hr_sr, x="ds_factor", y=metric, hue="source", palette="pastel", dodge=True)
        sns.stripplot(data=df_hr_sr, x="ds_factor", y=metric, hue="source",
                      dodge=True, color="black", alpha=0.3, jitter=0.2, size=2, legend=False)

        if draw_lines:
            hue_order = ["HR", "SR"]
            hue_offsets = {"HR": -0.2, "SR": 0.2}
            positions_by_ds_source = {
                ds: {source: i + hue_offsets[source] for source in hue_order}
                for i, ds in enumerate(sorted(df_hr_sr["ds_factor"].unique()))
            }

            df_pivot = df_hr_sr.pivot_table(index=["ds_factor", "patch_idx"], columns="source", values=metric)
            for (ds, patch_idx), row in df_pivot.dropna().iterrows():
                x_hr = positions_by_ds_source[ds]["HR"]
                x_sr = positions_by_ds_source[ds]["SR"]
                y_hr = row["HR"]
                y_sr = row["SR"]
                ax.plot([x_hr, x_sr], [y_hr, y_sr], color="gray", alpha=0.3, linewidth=0.8)
                ax.scatter([x_hr, x_sr], [y_hr, y_sr], color="black", s=10, alpha=0.5)

        plt.title(metric.replace("_", " ").title() + " (HR vs SR only)")
        plt.xlabel("Downsampling Factor")
        ylabel = get_ylabel(metric)
        plt.ylabel(ylabel)
        plt.legend(title="Source", loc="upper right")
        plt.tight_layout()
        fname = f"{metric}_HR_SR_by_ds"
        if draw_lines:
            fname += "_lines"
        plt.savefig(os.path.join(box_hrsr_dir, f"{fname}.png"))
        plt.close()

    # ==================== All Sources - VIOLIN (no lines) ====================
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.violinplot(data=df_all_sources, x="ds_factor", y=metric, hue="source", palette="pastel", inner=None)
    sns.stripplot(data=df_all_sources, x="ds_factor", y=metric, hue="source",
                  dodge=True, color="black", alpha=0.3, jitter=0.2, size=2, legend=False)
    plt.title(metric.replace("_", " ").title())
    plt.xlabel("Downsampling Factor")
    ylabel = get_ylabel(metric)
    plt.ylabel(ylabel)
    plt.legend(title="Source", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(violin_dir, f"{metric}_HR_LR_SR_violin_by_ds.png"))
    plt.close()

    # ==================== HR vs SR Only - VIOLIN (no lines) ====================
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.violinplot(data=df_hr_sr, x="ds_factor", y=metric, hue="source", palette="pastel", inner=None)
    sns.stripplot(data=df_hr_sr, x="ds_factor", y=metric, hue="source",
                  dodge=True, color="black", alpha=0.3, jitter=0.2, size=2, legend=False)
    plt.title(metric.replace("_", " ").title() + " (HR vs SR Violin)")
    plt.xlabel("Downsampling Factor")
    ylabel = get_ylabel(metric)
    plt.ylabel(ylabel)
    plt.legend(title="Source", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(violin_hrsr_dir, f"{metric}_HR_SR_violin_by_ds.png"))
    plt.close()

print(f"All plots saved in '{output_dir}'.")


# ======================== Test normality with wilcoxon test ==================

# Store results
normality_results = []
wilcoxon_results = []

for metric in metric_cols:
    for ds in DS_FACTORS:
        # Filter to current downsampling factor and HR/SR
        df_ds = df[(df["ds_factor"] == ds) & (df["source"].isin(["HR", "SR"]))]
        
        # Pivot to get HR and SR per patch
        df_pivot = df_ds.pivot_table(index="patch_idx", columns="source", values=metric)
        df_paired = df_pivot.dropna()

        # Skip if too few pairs
        if len(df_paired) < 3:
            print(f"Skipping metric '{metric}', ds={ds} due to too few samples.")
            continue

        # Compute paired differences
        diffs = df_paired["HR"] - df_paired["SR"]

        # --- Shapiro-Wilk normality test ---
        try:
            shapiro_stat, shapiro_p = shapiro(diffs)
        except Exception as e:
            shapiro_stat, shapiro_p = float("nan"), float("nan")
            print(f"Shapiro test failed for metric '{metric}', ds={ds}: {e}")

        normality_results.append({
            "metric": metric,
            "ds_factor": ds,
            "n_patches": len(diffs),
            "shapiro_stat": shapiro_stat,
            "shapiro_p_value": shapiro_p
        })

        # --- Wilcoxon signed-rank test ---
        try:
            wilcoxon_stat, wilcoxon_p = wilcoxon(df_paired["HR"], df_paired["SR"])
        except Exception as e:
            wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
            print(f"Wilcoxon test failed for metric '{metric}', ds={ds}: {e}")

        wilcoxon_results.append({
            "metric": metric,
            "ds_factor": ds,
            "n_patches": len(diffs),
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_p_value": wilcoxon_p
        })

# --- Convert to DataFrames ---
normality_df = pd.DataFrame(normality_results)
wilcoxon_df = pd.DataFrame(wilcoxon_results)

# ======================== Bland–Altman Plots ========================

def bland_altman_plot(df, metric, ds_factor, save_path=None):
    # Filter for HR and SR at a specific downsampling factor
    df_hr_sr = df[(df["ds_factor"] == ds_factor) & (df["source"].isin(["HR", "SR"]))]

    # Pivot to pair HR and SR values per patch
    df_pivot = df_hr_sr.pivot_table(index="patch_idx", columns="source", values=metric)
    df_paired = df_pivot.dropna()

    if len(df_paired) < 3:
        print(f"Skipping Bland–Altman for {metric}, ds={ds_factor} (too few pairs)")
        return

    hr = df_paired["HR"]
    sr = df_paired["SR"]
    mean_vals = (hr + sr) / 2

    if metric == "bone_volume_fraction":
        diff_vals = sr - hr
        ylabel = 'Difference (SR − HR) [%]'
        xlabel = 'Mean of HR and SR [%]'
    else:
        diff_vals = sr - hr
        ylabel = 'Difference (SR − HR)'
        xlabel = 'Mean of HR and SR'


    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)

    # Plot
    plt.figure(figsize=(8, 6), dpi=150)
    sns.scatterplot(x=mean_vals, y=diff_vals, alpha=0.6, edgecolor='k', s=30)

    # Bias and limits of agreement
    if metric == "bone_volume_fraction":
        legend_unit = "%"
    else:
        legend_unit = "mm"  # or specify mm, μm, etc. if known

    plt.axhline(mean_diff, color='blue', linestyle='--',
                label=f'Mean diff: {mean_diff:.2f}{legend_unit}')
    plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--',
                label=f'+1.96 SD: {(mean_diff + 1.96 * std_diff):.2f}{legend_unit}')
    plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--',
                label=f'−1.96 SD: {(mean_diff - 1.96 * std_diff):.2f}{legend_unit}')


    plt.title(f'Bland–Altman: {metric.replace("_", " ").title()} (DS={ds_factor})')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()

# Create output folder
bland_dir = os.path.join(output_dir, "bland_altman_plots")
os.makedirs(bland_dir, exist_ok=True)

# Generate Bland–Altman plots for all metric × ds combinations
for metric in metric_cols:
    for ds in DS_FACTORS:
        save_path = os.path.join(bland_dir, f"bland_altman_{metric}_ds{ds}.png")
        bland_altman_plot(df, metric, ds, save_path)


def remove_outliers_iqr(series, multiplier=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return series[(series >= lower) & (series <= upper)]

def bland_altman_plot(df, metric, ds_factor, save_path=None, remove_outliers=False):
    df_hr_sr = df[(df["ds_factor"] == ds_factor) & (df["source"].isin(["HR", "SR"]))]
    df_pivot = df_hr_sr.pivot_table(index="patch_idx", columns="source", values=metric)
    df_paired = df_pivot.dropna()

    if len(df_paired) < 3:
        print(f"Skipping Bland–Altman for {metric}, ds={ds_factor} (too few pairs)")
        return

    hr = df_paired["HR"]
    sr = df_paired["SR"]

    # Values for statistics
    mean_vals_all = (hr + sr) / 2
    diff_vals_all = sr - hr

    # Values for plotting
    if remove_outliers:
        inliers = remove_outliers_iqr(diff_vals_all)
        mean_vals_plot = mean_vals_all.loc[inliers.index]
        diff_vals_plot = diff_vals_all.loc[inliers.index]
    else:
        mean_vals_plot = mean_vals_all
        diff_vals_plot = diff_vals_all

    # Use all values for stats (even if some are not plotted)
    mean_diff = np.mean(diff_vals_all)
    std_diff = np.std(diff_vals_all)

    # Units
    if metric == "bone_volume_fraction":
        ylabel = 'Difference (SR − HR) [%]'
        xlabel = 'Mean of HR and SR [%]'
        legend_unit = "%"
    else:
        ylabel = 'Difference (SR − HR)'
        xlabel = 'Mean of HR and SR'
        legend_unit = "mm"  # or µm if appropriate

    # Plot
    plt.figure(figsize=(8, 6), dpi=150)
    sns.scatterplot(x=mean_vals_plot, y=diff_vals_plot, alpha=0.6, edgecolor='k', s=30)

    plt.axhline(mean_diff, color='blue', linestyle='--',
                label=f'Mean diff: {mean_diff:.2f}{legend_unit}')
    plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--',
                label=f'+1.96 SD: {(mean_diff + 1.96 * std_diff):.2f}{legend_unit}')
    plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--',
                label=f'−1.96 SD: {(mean_diff - 1.96 * std_diff):.2f}{legend_unit}')

    title_suffix = " (No Outliers)" if remove_outliers else ""
    plt.title(f'Bland–Altman: {metric.replace("_", " ").title()} (DS={ds_factor}){title_suffix}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if metric in ["trabecular_thickness", "trabecular_thickness_mean"]:
        plt.xlim(0, 0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


# Create separate output folders
bland_dir_raw = os.path.join(output_dir, "bland_altman_plots")
bland_dir_clean = os.path.join(output_dir, "bland_altman_plots_no_outliers")
os.makedirs(bland_dir_raw, exist_ok=True)
os.makedirs(bland_dir_clean, exist_ok=True)

# Generate both versions
for metric in metric_cols:
    for ds in DS_FACTORS:
        fname_raw = f"bland_altman_{metric}_ds{ds}.png"
        fname_clean = f"bland_altman_{metric}_ds{ds}_no_outliers.png"
        path_raw = os.path.join(bland_dir_raw, fname_raw)
        path_clean = os.path.join(bland_dir_clean, fname_clean)

        bland_altman_plot(df, metric, ds_factor=ds, save_path=path_raw, remove_outliers=False)
        bland_altman_plot(df, metric, ds_factor=ds, save_path=path_clean, remove_outliers=True)


def remove_outliers_df(df, metric, multiplier=1.5):
    def iqr_clip(group):
        q1 = group[metric].quantile(0.25)
        q3 = group[metric].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        return group[(group[metric] >= lower) & (group[metric] <= upper)]

    return df.groupby(["ds_factor", "source"], group_keys=False).apply(iqr_clip)

# Output dirs (no outliers)
output_dir_clean = "boxplots_across_ds_otsu_no_outliers"
box_dir_clean = os.path.join(output_dir_clean, "boxplots_all_sources")
box_hrsr_dir_clean = os.path.join(output_dir_clean, "boxplots_hr_sr_only")
violin_dir_clean = os.path.join(output_dir_clean, "violinplots_all_sources")
violin_hrsr_dir_clean = os.path.join(output_dir_clean, "violinplots_hr_sr_only")

for d in [box_dir_clean, box_hrsr_dir_clean, violin_dir_clean, violin_hrsr_dir_clean]:
    os.makedirs(d, exist_ok=True)

for metric in metric_cols:
    df_all_sources_clean = remove_outliers_df(df, metric)
    df_hr_sr_clean = df_all_sources_clean[df_all_sources_clean["source"].isin(["HR", "SR"])]

    for draw_lines in [False, True]:
        # --- BOXPLOT: All Sources ---
        plt.figure(figsize=(10, 6), dpi=300)
        ax = sns.boxplot(data=df_all_sources_clean, x="ds_factor", y=metric, hue="source", palette="pastel")
        sns.stripplot(data=df_all_sources_clean, x="ds_factor", y=metric, hue="source",
                      dodge=True, color="black", alpha=0.3, jitter=0.2, size=2, legend=False)

        plt.title(metric.replace("_", " ").title() + " (No Outliers)")
        plt.xlabel("Downsampling Factor")
        ylabel = get_ylabel(metric)
        plt.ylabel(ylabel)
        plt.legend(title="Source", loc="upper right")
        plt.tight_layout()
        fname = f"{metric}_HR_LR_SR_by_ds_no_outliers"
        if draw_lines:
            fname += "_lines"
        plt.savefig(os.path.join(box_dir_clean, f"{fname}.png"))
        plt.close()

        # --- BOXPLOT: HR vs SR only ---
        plt.figure(figsize=(10, 6), dpi=300)
        ax = sns.boxplot(data=df_hr_sr_clean, x="ds_factor", y=metric, hue="source", palette="pastel", dodge=True)
        sns.stripplot(data=df_hr_sr_clean, x="ds_factor", y=metric, hue="source",
                      dodge=True, color="black", alpha=0.3, jitter=0.2, size=2, legend=False)

        plt.title(metric.replace("_", " ").title() + " (HR vs SR only, No Outliers)")
        plt.xlabel("Downsampling Factor")
        ylabel = get_ylabel(metric)
        plt.ylabel(ylabel)
        plt.legend(title="Source", loc="upper right")
        plt.tight_layout()
        fname = f"{metric}_HR_SR_by_ds_no_outliers"
        if draw_lines:
            fname += "_lines"
        plt.savefig(os.path.join(box_hrsr_dir_clean, f"{fname}.png"))
        plt.close()

    # --- VIOLIN: All Sources ---
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.violinplot(data=df_all_sources_clean, x="ds_factor", y=metric, hue="source", palette="pastel", inner=None)
    sns.stripplot(data=df_all_sources_clean, x="ds_factor", y=metric, hue="source",
                  dodge=True, color="black", alpha=0.3, jitter=0.2, size=2, legend=False)

    plt.title(metric.replace("_", " ").title() + " (No Outliers)")
    plt.xlabel("Downsampling Factor")
    ylabel = get_ylabel(metric)
    plt.ylabel(ylabel)
    plt.legend(title="Source", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(violin_dir_clean, f"{metric}_HR_LR_SR_violin_by_ds_no_outliers.png"))
    plt.close()

    # --- VIOLIN: HR vs SR only ---
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.violinplot(data=df_hr_sr_clean, x="ds_factor", y=metric, hue="source", palette="pastel", inner=None)
    sns.stripplot(data=df_hr_sr_clean, x="ds_factor", y=metric, hue="source",
                  dodge=True, color="black", alpha=0.3, jitter=0.2, size=2, legend=False)

    plt.title(metric.replace("_", " ").title() + " (HR vs SR Violin, No Outliers)")
    plt.xlabel("Downsampling Factor")
    ylabel = get_ylabel(metric)
    plt.ylabel(ylabel)
    plt.legend(title="Source", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(violin_hrsr_dir_clean, f"{metric}_HR_SR_violin_by_ds_no_outliers.png"))
    plt.close()

print(f"\n All outlier-free box/violin plots saved in '{output_dir_clean}'")



# --- Print results ---

def format_p(p):
    return f"{p:.2e}" if p < 0.01 else f"{p:.3f}"

print("\n=== Wilcoxon Test Summary (HR vs SR) ===")
for row in wilcoxon_df.itertuples(index=False):
    sig = "✔️" if row.wilcoxon_p_value < 0.05 else "✘"
    print(f"{row.metric:<30} ds={row.ds_factor:<2}  p={format_p(row.wilcoxon_p_value):<10}  significant: {sig}")

print("\n=== Shapiro-Wilk Test Summary (Normality of HR - SR) ===")
for row in normality_df.itertuples(index=False):
    norm = "✔️" if row.shapiro_p_value > 0.05 else "✘"
    print(f"{row.metric:<30} ds={row.ds_factor:<2}  p={format_p(row.shapiro_p_value):<10}  normal: {norm}")

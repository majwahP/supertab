import pandas as pd

# Load data
df = pd.read_csv("stat_outputs/trabecular_metrics_ds4_200ep.csv")

# Add ds_factor manually if missing
df["ds_factor"] = 4

# Keep only HR and SR rows
df = df[df["source"].isin(["HR", "SR"])]

# Select only metric columns (numeric types)
exclude_cols = {"position", "source", "patch_idx", "ds_factor"}
metric_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

# Keep only necessary columns for pivoting
df_pivotable = df[["patch_idx", "source"] + metric_cols]

# Pivot to get HR and SR values on same row
df_pivot = df_pivotable.pivot(index="patch_idx", columns="source", values=metric_cols)

# Compute MAE for each metric
mae_results = {}
for metric in metric_cols:
    sr_vals = df_pivot[(metric, "SR")]
    hr_vals = df_pivot[(metric, "HR")]
    mae = (sr_vals - hr_vals).abs().mean()
    mae_results[metric] = mae

# Print results
print("Mean Absolute Error (SR vs HR):")
for metric, mae in mae_results.items():
    print(f"{metric}: MAE = {mae:.4f}")

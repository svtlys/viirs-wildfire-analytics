import pandas as pd

# Load your full dataset
# Replace with your actual file name
df = pd.read_csv("fire_archive_J1V-C2_726206.csv")

# Keep rows that help tell the story better:
# 1. nominal + high confidence fires only
# 2. rows with complete values in important columns
# 3. FRP threshold to remove very weak detections
df_filtered = (
    df[df["confidence"].isin(["n", "h"])]
    .dropna(subset=["latitude", "longitude", "frp", "brightness", "bright_t31"])
)

# keep only stronger fires
df_filtered = df_filtered[df_filtered["frp"] > 10]

# Optional: remove extreme outliers if needed
# Uncomment this only if you decide the highest FRP values distort the plots
# df_filtered = df_filtered[df_filtered["frp"] < 500]

# Keep only columns needed for the dashboards
cols = [
    "latitude",
    "longitude",
    "acq_date",
    "brightness",
    "bright_t31",
    "frp",
    "confidence"
]

df_filtered = df_filtered[cols]

# Sample rows for dashboard performance
sample_size = min(40000, len(df_filtered))
df_dashboard = df_filtered.sample(n=sample_size, random_state=42)

# Save to CSV
df_dashboard.to_csv("data/dashboard_dataset.csv", index=False)

print("Original rows:", len(df))
print("Filtered rows:", len(df_filtered))
print("Sampled rows saved:", len(df_dashboard))
print("Saved file: data/dashboard_dataset.csv")

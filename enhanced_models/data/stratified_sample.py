import pandas as pd

# -----------------------------
# Settings
# -----------------------------
input_file = "fires.csv"
output_file = "stratified_sample.csv"
sample_fraction = 0.10   # Adjust this (e.g., 0.05, 0.2)
random_seed = 42

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(input_file)

# -----------------------------
# Convert date column
# -----------------------------
df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
df = df.dropna(subset=["acq_date"]).copy()

# -----------------------------
# Extract month for stratification
# -----------------------------
df["month"] = df["acq_date"].dt.month

# -----------------------------
# Proportional stratified sampling
# (same fraction from each month → preserves proportions)
# -----------------------------
sampled_df = (
    df.groupby("month", group_keys=False)
      .apply(lambda x: x.sample(frac=sample_fraction, random_state=random_seed))
      .reset_index(drop=True)
)

# -----------------------------
# Check that proportions are preserved
# -----------------------------
original_dist = df["month"].value_counts(normalize=True).sort_index()
sampled_dist = sampled_df["month"].value_counts(normalize=True).sort_index()

print("Original month proportions:\n", original_dist)
print("\nSampled month proportions:\n", sampled_dist)

# -----------------------------
# Drop helper column
# -----------------------------
sampled_df = sampled_df.drop(columns=["month"])

# -----------------------------
# Save sampled dataset
# -----------------------------
sampled_df.to_csv(output_file, index=False)

print("\nSaved to:", output_file)
print("Original size:", len(df))
print("Sampled size:", len(sampled_df))
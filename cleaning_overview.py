import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "DL_FIRE_J1V-C2_725729/fire_nrt_J1V-C2_725729.csv"

df = pd.read_csv(file_path)

# This file is intended to understand the dataset data and clean data
# Note that there are no nulls and that the dataset is already quite clean 

# ----------------------------
# Basic Overview
# ----------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

# ----------------------------
# Check for null values 
# ----------------------------

# Note that this dataset doesn't have any null values 
print("\nMissing values:")
print(df.isnull().sum())



# ----------------------------
# 4. Quick Statistical Summary
# ----------------------------

print("\nSummary statistics (numeric columns):")
print(df.describe())



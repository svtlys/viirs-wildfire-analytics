# 🔥 VIIRS Wildfire Analytics Project

## Project Overview

This project analyzes global wildfire activity using NASA VIIRS satellite data from 2024–2025. Our goal is to identify spatial and temporal wildfire patterns and explore how well machine learning models can predict wildfire occurrences.

We combine GPU-accelerated data processing (RAPIDS), exploratory data analysis (EDA), and interactive dashboards to uncover insights and support future predictive modeling.

---

## Research Question

**What wildfire patterns can be identified from VIIRS satellite data between 2024 and 2025, and how well can a machine learning model trained on historical data predict wildfire occurrences compared to actual events in 2025?**

---

## 📊 Dataset

* **Source:** NASA VIIRS (Visible Infrared Imaging Radiometer Suite)  
* **Time Range:** 2024–2025  
* **Size:** ~22 million records  

### Access Instructions

The dataset was downloaded from the source and sampled. All intermediary data is stored externally on Google Drive due to its size:  
👉 https://drive.google.com/file/d/17lHVkabbMYZ8FH9FFHpUXV9TBRMjlCPW/view?usp=sharing  

**Steps:**
1. Download the dataset  
2. Place it inside the `data/` folder in this repository  

---

## ⚙️ Data Processing (RAPIDS)

We used NVIDIA RAPIDS to accelerate data processing on the GPU.

### Key Steps

* Converted data into GPU DataFrames (cuDF)  
* Parsed and transformed date fields  
* Created a **"period" feature** (bi-monthly seasonal grouping)  
* Filtered dataset to include **nominal and high-confidence fires only**  
* Performed large-scale aggregations (groupby operations)  

### Why RAPIDS?

* Significantly faster than CPU-based pandas  
* Enabled processing of tens of millions of rows efficiently  
* Reduced runtime from seconds/minutes to milliseconds for key operations  

---

## 🧹 Data Cleaning & Preparation

* Removed low-confidence detections to reduce noise  
* Converted date fields into datetime format  
* Engineered time-based features (month, seasonal period)  
* Aggregated spatial data into latitude/longitude bins  

### Challenges

* Handling large dataset size without crashing sessions  
* GPU memory limitations  
* Ensuring consistency when converting between cuDF and pandas  

---

## 📈 Exploratory Data Analysis (EDA)

We created multiple visualizations to explore wildfire behavior.

### Key Visualizations

* Global wildfire hotspot map  
* Fire detections over time (trend analysis)  
* Seasonal wildfire activity (period-based bar chart)  
* Fire intensity (FRP) distribution  
* Spatial density heatmap  
* Confidence vs. fire occurrence  

---

## 🔍 Key Insights

### 1. Granularity Matters
Monthly aggregation is insufficient to capture wildfire trends.  
Daily-level data reveals meaningful variation that is otherwise lost.  

### 2. Weak Feature Correlation
There is no strong linear correlation between features and FRP.  
This limits model performance and suggests complex, non-linear relationships.  

### 3. Data Ambiguity
Error distributions of fire classes show high variability and outliers.  
This ambiguity contributes to model error and reflects limitations in the dataset.  

### 4. Target Transformation Matters
Log transformation of FRP significantly improves model performance, especially for simpler models like linear regression.  
This indicates the original target distribution is highly skewed and benefits from normalization.  

---

## 🤖 Modeling

### 📁 Project Structure

* `baseline/` → Baseline modeling (minimal feature engineering)  
* `enhanced_models/` → Models with feature engineering applied  
* `PCA/` → PCA exploration conducted prior to modeling  

---

### ⚙️ Modeling Strategy

**Feature Engineering**
* Log transform FRP (handle skew)  
* Extract month (capture seasonality)  
* Encode categorical variables  
* Remove low-value features  

**Objective**
* Evaluate impact of:
  * Feature engineering  
  * Model complexity  
  * Hyperparameter tuning  

**Modeling Approach**
* Compare baseline vs enhanced models  
* Linear vs tree-based methods  
* Apply Grid Search for tuning  
* Use PCA in selected pipelines  
* Stratified train-test split to preserve target distribution  

---

## 📊 Dashboards

We developed **four dashboards using different tools**:

* **Tableau** – Visual analytics & correlations  
  👉 https://public.tableau.com/app/profile/angela.wei/viz/WildfireHotspotTemperatureandBrightnessDistributions/Dashboard  

* **Plotly Dash** – Interactive spatial & temporal analysis  

* **Power BI** – Business-style reporting dashboard  

* **Streamlit** – Lightweight interactive app  

All dashboard implementations can be found in the `dashboards/` folder.

---
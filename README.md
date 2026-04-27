# 🔥 VIIRS Wildfire Analytics Project

## 📌 Project Overview

This project analyzes global wildfire activity using NASA VIIRS satellite data from 2024–2025. Our goal is to identify spatial and temporal wildfire patterns and explore how well machine learning models can predict wildfire occurrences.

We combine GPU-accelerated data processing (RAPIDS), exploratory data analysis (EDA), and interactive dashboards to uncover insights and support future predictive modeling.

---

## ❓ Research Question

**What wildfire patterns can be identified from VIIRS satellite data between 2024 and 2025, and how well can a machine learning model trained on historical data predict wildfire occurrences compared to actual events in 2025?**

---

## 📊 Dataset

* **Source:** NASA VIIRS (Visible Infrared Imaging Radiometer Suite)
* **Time Range:** 2024–2025
* **Size:** ~22 million records 
* **Key Features:**

  * `acq_date` – Date of fire detection
  * `latitude`, `longitude` – Geographic location
  * `frp` – Fire Radiative Power (intensity)
  * `confidence` – Detection confidence (low, nominal, high)

### Why This Dataset?

* Real-world, large-scale satellite data
* High volume enables meaningful pattern detection
* Rich features allow both EDA and machine learning applications
* Relevant to climate and environmental monitoring

### Dataset Complexity

* Large size required GPU acceleration
* Temporal and spatial preprocessing needed
* Noise reduction required (confidence filtering)

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

### Key Insights

* Wildfires show strong seasonal patterns
* Certain regions exhibit consistently high fire activity
* Fire intensity varies significantly across time periods
* High-confidence fires provide cleaner analytical signals

---

## 🤖 Preliminary Machine Learning Direction

* **Task Type:** Classification or regression
* **Goal:** Predict wildfire occurrence or intensity

### Justification

* Strong temporal and seasonal trends observed in EDA
* Spatial clustering suggests predictive geographic patterns
* Cleaned dataset supports supervised learning

---

## 📊 Dashboards

We developed **two dashboards using different tools**:

### 1. Plotly Dash Dashboard

* Focus: Spatial & Temporal Analysis
* Features:

  * Interactive wildfire hotspot map
  * Time-series trends
  * Seasonal filtering
  * Dynamic user interaction

### 2. Tableau Dashboard

* Focus: Fire Intensity Analysis
* Features:

  * FRP distribution
  * Correlation visualizations
  * Comparative analysis across features
  * Dashboard: https://public.tableau.com/app/profile/angela.wei/viz/WildfireHotspotTemperatureandBrightnessDistributions/Dashboard 

---


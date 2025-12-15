**Project Title**

A Machine Learning Framework for Galaxy Morphology Classification, Redshift Estimation, and AGN Identification Using SDSS Data

**Project Overview**

The Sloan Digital Sky Survey (SDSS) provides large-scale astronomical data that enables the study of galaxy properties and cosmic evolution.
This project applies machine learning techniques to SDSS photometric data to solve three key astrophysical problems:

**Galaxy Morphology Classification

Galaxy Redshift Estimation

Active Galactic Nuclei (AGN) Identification**

The trained models are deployed through a Flask-based web application that allows users to interactively obtain predictions using galaxy parameters.

**Dataset Description**

The dataset is sourced from the Sloan Digital Sky Survey (SDSS) and contains approximately 100,000 galaxy observations with photometric and derived features.

Dataset link: https://www.kaggle.com/datasets/bryancimo/sdss-galaxy-classification-dr18

**Key Characteristics:**

Photometric magnitudes: u, g, r, i, z

Flux and radius-based features

Shape and axis ratio parameters

Redshift and redshift error values

Galaxy subclass labels (STARFORMING, STARBURST)

**Tools & Technologies Used**

**Programming & Libraries**

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

XGBoost

Joblib

**Web Framework**

Flask

HTML, CSS

**Development Environment**

Google Colab

Visual Studio Code

GitHub

**Data Preprocessing & Exploratory Data Analysis (EDA)**

The following preprocessing and EDA steps were performed:

Dataset loading and inspection

Replacement of invalid values (-9999) with NaN

Visualization of missing values using bar charts and heatmaps

Median imputation using SimpleImputer

Removal of rows with remaining missing values

Statistical analysis of skewness and kurtosis

Outlier treatment using Interquartile Range (IQR) clipping

Correlation analysis using heatmaps

These steps ensured data consistency and improved model robustness.

**Methodology & Scenarios**

**Scenario 1: Galaxy Morphology Classification**

**Objective:** Classify galaxies into morphological types such as Spiral, Elliptical, and Irregular.

Feature engineering using color indices (u-g, g-r, r-i, i-z)

Structural features such as Petrosian radii and axis ratios

Feature scaling using StandardScaler

Unsupervised clustering using K-Means

Cluster labeling and supervised classification

Final tuned model saved as: SDSS_S1_USC_LRC.pkl

**Scenario 2: Galaxy Redshift Estimation**

**Objective:** Predict galaxy redshift using photometric properties.

Removal of target leakage features

Selection of important photometric features

Missing feature handling using median statistics

Model training using XGBoost Regressor

Hyperparameter tuning and cross-validation

**Saved artifacts:**
SDSS_S2_Reg_XGB.pkl
SDSS_feature_medians.pkl
SDSS_feature_columns.pkl

**Scenario 3: AGN Identification**

**Objective:** Identify galaxies hosting Active Galactic Nuclei (AGN).

Color-based feature engineering

Compactness features using PSF and model flux differences

Flux ratio features across bands

Feature scaling and clustering

Supervised classification using XGBoost

Final model saved as: SDSS_S3_USC_XGB.pkl

**Web Application Deployment**

A Flask-based web application was developed to demonstrate real-time predictions for all three scenarios.

**Application Features:**

Interactive home page with SDSS imagery

Manual input-based predictions

Separate interfaces for each scenario

Clean and responsive UI

**Results**

Stable cross-validation performance across all scenarios

Successful deployment of trained ML models

Interactive prediction capability through web interface

**Conclusion**

This project demonstrates the effective application of machine learning techniques on large-scale astronomical data. By integrating data preprocessing, feature engineering, model optimization, and web deployment, the system provides an end-to-end solution for galaxy analysis using SDSS data.

🚀 Conclusion

This project demonstrates the effective application of machine learning techniques on large-scale astronomical data. By integrating data preprocessing, feature engineering, model optimization, and web deployment, the system provides an end-to-end solution for galaxy analysis using SDSS data.

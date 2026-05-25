# House Rent Prediction Using Linear Regression

An end-to-end machine learning regression project that predicts house rent based on property features such as size, BHK count, city, furnishing status, and tenant preferences.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Pipeline](#project-pipeline)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Building and Training](#model-building-and-training)
- [Model Evaluation](#model-evaluation)
- [Model Saving and Inference](#model-saving-and-inference)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Key Design Decisions](#key-design-decisions)

---

## Project Overview

This project implements a complete supervised machine learning pipeline to predict monthly house rent in Indian cities. Linear Regression is used as the core algorithm, with careful attention to regression assumptions — including linearity, multicollinearity, and residual autocorrelation — throughout the workflow.

The target variable (`Rent`) is log-transformed before training to handle skewness and improve the linear relationship with input features. Predictions are reverse-transformed using `expm1` to return rent values in the original scale.

---

## Dataset

**File:** `House_Rent_Dataset.csv`

The dataset contains listings of rental properties across multiple Indian cities with attributes describing the property and rental terms.

### Features Used After Preprocessing

| Feature            | Type        | Description                                                  |
|--------------------|-------------|--------------------------------------------------------------|
| BHK                | Numerical   | Number of Bedrooms, Hall, and Kitchen                        |
| Size               | Numerical   | Property size in square feet                                 |
| City               | Categorical | City where the property is located                           |
| Area Type          | Categorical | Super Area, Carpet Area, or Built Area                       |
| Furnishing Status  | Categorical | Furnished, Semi-Furnished, or Unfurnished                    |
| Tenant Preferred   | Categorical | Bachelors, Family, or Bachelors/Family                       |
| Point of Contact   | Categorical | Contact Owner, Contact Agent, or Contact Builder             |

### Columns Dropped

| Column         | Reason for Removal                                                      |
|----------------|-------------------------------------------------------------------------|
| Floor          | High cardinality, inconsistent formatting                               |
| Area Locality  | Too granular; leads to sparse encoding                                  |
| Posted On      | Date field not relevant to rent price modeling                          |
| Bathroom       | Dropped after VIF analysis revealed multicollinearity with BHK and Size |

### Target Variable

- **Raw:** `Rent` (monthly rent in INR)
- **Transformed:** `log_rent` = `log(Rent)` — used for model training
- **Prediction output:** Reverse-transformed via `np.expm1(log_rent)`

---

## Tech Stack

| Library        | Purpose                                              |
|----------------|------------------------------------------------------|
| pandas         | Data loading, manipulation, and feature engineering  |
| numpy          | Numerical operations and log transformation          |
| matplotlib     | Static visualizations                                |
| seaborn        | Statistical plots and styled visualizations          |
| scikit-learn   | Train-test split, scaling, model training, metrics   |
| statsmodels    | VIF computation, Durbin-Watson test                  |
| joblib         | Saving and loading trained model and scaler          |

---

## Project Pipeline

```
Raw CSV Data
     |
     v
Data Loading & Inspection
     |
     v
Exploratory Data Analysis (EDA)
     |
     v
Outlier Removal
     |
     v
Log Transformation of Target
     |
     v
Feature Selection (VIF Check)
     |
     v
One-Hot Encoding of Categorical Features
     |
     v
Train-Test Split (80/20)
     |
     v
Feature Scaling (StandardScaler)
     |
     v
Linear Regression Model Training
     |
     v
Model Evaluation (R2, MAE, RMSE, Residual Analysis)
     |
     v
Reverse Transformation & Prediction
     |
     v
Model Serialization (joblib)
```

---

## Exploratory Data Analysis

The EDA phase examines both the distribution of individual features and their relationship with the target variable.

**Numerical distributions** — Histograms with KDE curves are plotted for `Rent`, `Size`, `BHK`, and `Bathroom` to understand spread and skewness.

**Categorical distributions** — Count plots are generated for `City`, `Area Type`, `Furnishing Status`, `Tenant Preferred`, and `Point of Contact`.

**Rent vs numerical features** — Scatter plots of `Rent` against `Size`, `BHK`, and `Bathroom` are used to assess linearity before and after log transformation.

**Rent vs categorical features** — Box plots show how rent varies across categories, helping identify which categorical groups have the most pricing power.

**Correlation heatmap** — A heatmap of the numeric feature correlation matrix is used to detect pairwise relationships and potential multicollinearity.

---

## Data Preprocessing

### Outlier Removal

Outliers are capped at the 99th percentile separately for `Rent` and `Size` to reduce the influence of extreme values without discarding too much data.

```python
rent_upper_limit = df["Rent"].quantile(0.99)
df = df[df["Rent"] < rent_upper_limit]

size_upper_limit = df["Size"].quantile(0.99)
df = df[df["Size"] < size_upper_limit]
```

### Log Transformation

The `Rent` column is right-skewed. Applying a log transformation produces a more normally distributed target and improves the linear fit.

```python
df["log_rent"] = np.log(df["Rent"])
```

### Multicollinearity Check (VIF)

Variance Inflation Factor (VIF) is computed for `BHK` and `Size` after `Bathroom` is removed. Low VIF scores confirm that multicollinearity is not a concern for the remaining numerical features.

### One-Hot Encoding

All remaining categorical columns are encoded using `pd.get_dummies` with `drop_first=True` to avoid the dummy variable trap, and output as integer columns.

---

## Model Building and Training

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Scaling** — `StandardScaler` is applied to `BHK` and `Size`. The scaler is fit only on the training set and then applied to the test set to prevent data leakage.

**Model** — `sklearn.linear_model.LinearRegression` is trained on the scaled training data.

---

## Model Evaluation

The model is assessed using four approaches:

### Regression Metrics

| Metric     | Description                                                   |
|------------|---------------------------------------------------------------|
| R2 Score   | Proportion of variance in log_rent explained by the model     |
| MAE        | Mean absolute error on the log scale                          |
| RMSE       | Root mean squared error on the log scale                      |

### Actual vs Predicted Plot

A scatter plot of actual vs predicted `log_rent` values, overlaid with a perfect prediction line (y = x), shows how closely predictions track true values.

### Residual Analysis

- **Residual distribution** — A histogram with KDE checks whether residuals are approximately normally distributed and centered at zero.
- **Residuals vs predicted** — A scatter plot checks for homoscedasticity (constant variance of residuals across predicted values).

### Durbin-Watson Test

The Durbin-Watson statistic is computed to test for autocorrelation in residuals. A value close to 2.0 indicates no significant autocorrelation, confirming that the independence assumption holds.

---

## Model Saving and Inference

The trained model and scaler are saved for reuse without retraining:

```python
joblib.dump(model, "house_rent_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

### Sample Prediction

A sample prediction is demonstrated for a 2 BHK, 1200 sq. ft. Semi-Furnished property in Mumbai with the following steps:

1. Create a DataFrame with the raw input features
2. Apply `pd.get_dummies` and reindex to match training columns
3. Scale numerical features using the saved scaler
4. Predict `log_rent` using the saved model
5. Reverse-transform: `predicted_rent = np.expm1(model.predict(sample_data))`

---

## Project Structure

```
├── House_RentPrediction_project.ipynb   # Main notebook with full pipeline
├── House_Rent_Dataset.csv               # Input dataset
├── house_rent_model.pkl                 # Serialized trained Linear Regression model
├── scaler.pkl                           # Serialized StandardScaler
└── README.md                            # Project documentation
```

---

## How to Run

1. Clone or download the repository.

2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels joblib
   ```

3. Place `House_Rent_Dataset.csv` in the same directory as the notebook.

4. Open the notebook:
   ```bash
   jupyter notebook House_RentPrediction_project.ipynb
   ```

5. Run all cells in order. The notebook will generate all visualizations, print evaluation metrics, and save the model files to the working directory.

---

## Key Design Decisions

**Log transformation of target** — Raw rent values are heavily right-skewed. Log transformation normalizes the distribution and strengthens the linear relationship between features and the target, which is a core assumption of Linear Regression.

**99th percentile outlier capping** — Hard removal of extreme outliers in `Rent` and `Size` was preferred over IQR-based clipping to retain more data while still limiting the distortion caused by very high-value listings.

**Bathroom dropped via VIF** — Rather than relying on correlation alone, VIF was used to formally detect multicollinearity. `Bathroom` showed redundancy with `BHK` and `Size` and was removed to improve model stability.

**drop_first=True in encoding** — Avoids perfect multicollinearity introduced by dummy variables representing all levels of a categorical feature.

**Scaler fit only on train set** — Prevents data leakage by ensuring that test set statistics do not influence the scaling parameters used during training.

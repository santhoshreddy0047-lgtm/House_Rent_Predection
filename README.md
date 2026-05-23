# 🏠 House Rent Prediction Using Linear Regression

An end-to-end machine learning regression project that predicts house rent based on property features such as size, BHK count, city, furnishing status, and more.

---

## 📌 Project Overview

This project builds a Linear Regression model to predict house rent prices. It covers the full ML pipeline — from data loading and exploratory analysis to model training, evaluation, and saving the trained model for inference.

---

## 📂 Dataset

**File:** `House_Rent_Dataset.csv`

**Features used after preprocessing:**

| Feature | Type | Description |
|---|---|---|
| BHK | Numerical | Number of Bedrooms, Hall, Kitchen |
| Size | Numerical | Size of the property in sq. ft. |
| City | Categorical | City where the property is located |
| Area Type | Categorical | Type of area (Super Area, Carpet Area, etc.) |
| Furnishing Status | Categorical | Furnished / Semi-Furnished / Unfurnished |
| Tenant Preferred | Categorical | Preferred tenant type |
| Point of Contact | Categorical | Owner / Agent / Builder |

**Dropped columns:** `Floor`, `Area Locality`, `Posted On`, `Bathroom` (due to multicollinearity/low relevance)

**Target variable:** `Rent` (log-transformed to `log_rent` for modeling)

---

## 🔧 Tech Stack

- **Python 3**
- **pandas**, **numpy** — data manipulation
- **matplotlib**, **seaborn** — data visualization
- **scikit-learn** — model building, scaling, evaluation
- **statsmodels** — VIF analysis, Durbin-Watson test
- **joblib** — model serialization

---

## 🚀 Project Pipeline

1. **Data Loading** — Load dataset using pandas
2. **Exploratory Data Analysis (EDA)**
   - Distribution plots for numerical and categorical features
   - Scatter plots (Rent vs numerical features)
   - Box plots (Rent vs categorical features)
   - Correlation heatmap
3. **Data Preprocessing**
   - Drop irrelevant columns
   - Outlier removal using 99th percentile capping (Rent, Size)
   - Log transformation of target variable (`Rent` → `log_rent`)
   - VIF check to detect multicollinearity
   - One-hot encoding of categorical features
4. **Model Building**
   - 80/20 train-test split
   - StandardScaler for numerical features
   - Linear Regression model training
5. **Model Evaluation**
   - R² Score, MAE, RMSE
   - Actual vs Predicted plot
   - Residual analysis (distribution + scatter)
   - Durbin-Watson test for autocorrelation
6. **Prediction**
   - Reverse log transformation (`expm1`) to get rent in original scale
   - Sample prediction on new input data
7. **Model Saving**
   - `house_rent_model.pkl` — trained Linear Regression model
   - `scaler.pkl` — fitted StandardScaler

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| R² Score | Evaluated on test set |
| MAE | Mean Absolute Error on test set |
| RMSE | Root Mean Squared Error on test set |

> Run the notebook to see the exact metric values produced during training.

---

## 🗂️ Project Structure

```
├── House_RentPrediction_project.ipynb   # Main notebook
├── House_Rent_Dataset.csv               # Input dataset
├── house_rent_model.pkl                 # Saved trained model
├── scaler.pkl                           # Saved feature scaler
└── README.md                            # Project documentation
```

---

## ▶️ How to Run

1. Clone or download the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels joblib
   ```
3. Place `House_Rent_Dataset.csv` in the same directory as the notebook.
4. Open and run `House_RentPrediction_project.ipynb` in Jupyter Notebook or JupyterLab.

---

## 🔮 Sample Prediction

The notebook includes a sample prediction for a 2 BHK, 1200 sq. ft. Semi-Furnished property in Mumbai. After encoding, scaling, and reversing the log transformation, the model outputs a predicted monthly rent.

---

## 📝 Notes

- The `Bathroom` feature was dropped after VIF analysis indicated multicollinearity with `BHK` and `Size`.
- Log transformation of `Rent` was applied to reduce skewness and improve model fit.
- Outliers in `Rent` and `Size` were capped at the 99th percentile before modeling.

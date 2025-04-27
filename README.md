# 🛍 Rossmann Retail Sales Forecasting: Causal Impact Analysis and Explainable Machine Learning

---

## 📖 Project Overview

In this project, we develop a complete data science pipeline to forecast daily sales for a major retail chain (Rossmann), quantify the true business impact of promotional campaigns, and provide full explainability of model decisions using modern AI techniques like SHAP.

We combine **traditional time series forecasting**, **machine learning regression models**, **manual causal impact analysis**, and **SHAP explainability** into a professional, real-world forecasting solution.

---

## ❓ Problem Statement

Retailers like Rossmann face highly volatile daily sales influenced by promotions, competition, holidays, and store characteristics.  
Traditional forecasting models (like ARIMA or Prophet) can predict trends but **cannot quantify the business impact of promotions** or **explain why predictions are made**.

**Our goals:**
- Predict future sales accurately.
- Quantify how much promotions actually lift sales.
- Make machine learning model predictions interpretable to business users.

---

## 📚 Dataset Summary

**Rossmann Sales Dataset** from Kaggle:
- ~1 million rows of daily sales data (2013-2015).
- Includes store metadata (type, assortment, competition, promotions).
- Data for 1115 stores across different regions.

---

## 🏗 Project Architecture

```
rossmann_forecasting/
├── data/                      # Raw and cleaned datasets
├── notebooks/                  # Exploratory notebooks per step
├── src/                        # Modular Python scripts
├── outputs/                    # Saved plots, causal reports
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
```

---

## 🛠️ Approach and Methods

### 1. Data Preprocessing
- Merged sales and store metadata.
- Parsed dates into year, month, day, and week features.
- Removed closed stores and zero-sales records.

### 2. Feature Engineering
- Lag features (7, 14, 30 days).
- Rolling mean features (7-day, 30-day).
- Promo duration features (cumulative running promo days).
- Competition open time features (months since competitor store opened).

### 3. Baseline Statistical Modeling
- **Prophet** and **SARIMA** time series models built for each store.
- RMSE performance recorded.

### 4. Machine Learning Forecasting
- **Random Forest** and **XGBoost** regression models.
- Feature set: lagged sales, promotions, competition, seasonality.
- Evaluation metrics: **RMSE**, **MAE**.

### 5. Causal Impact Analysis (Manual)
- Built counterfactual sales model using linear regression on pre-promo period.
- Measured uplift during promotions by comparing predicted vs actual sales.
- Quantified average and relative lift in sales.

### 6. Model Explainability
- **SHAP values** for model interpretation (feature importance and individual impact).
- **Partial Dependence Plots (PDP)** for top features.

---

## 📊 Key Results

| Model | RMSE (Test Data) |
|:---|:---|
| Prophet (Baseline) | ~780 |
| SARIMAX | ~1285 |
| Random Forest Regressor | **357.14** |
| XGBoost Regressor | **368.47** |

✅ Machine learning models outperformed traditional baselines, reducing RMSE by ~50%.

---

## 🎯 Causal Impact Findings

- **Average Daily Sales Lift from Promotions**: **$39.96**
- **Relative Lift**: **+0.90%**
- Promotions showed a positive but moderate impact on sales volumes.

✅ Sales behavior during promotions was significantly higher than model-predicted counterfactual sales without promotions.

---

## 🖼️ Visualizations (Located in `/outputs/plots/`)

- **SHAP Summary Plot** (`shap_summary.png`): Global feature importance across sales forecasts.
- **Random Forest Feature Importance Plot** (`rf_feature_importance.png`): Top predictive features for sales.
- **Partial Dependence Plots** (`pdp_plots_rf.png`): Impact of Promo, Day of Month, and Sales history.
- **Causal Impact Plot** (`causalimpact_plot.png`): Actual vs Predicted Sales During Promotion.

---

## ⚙️ Technologies Used

| Tool | Purpose |
|:---|:---|
| Python 3 | Main programming language |
| pandas, numpy | Data manipulation |
| scikit-learn | Machine learning models |
| XGBoost | Gradient boosting |
| Prophet | Time series forecasting |
| statsmodels | ARIMA modeling |
| SHAP | Model explainability |
| Matplotlib, Seaborn | Data visualization |

---

## 🚀 How to Run This Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rossmann_forecasting.git
   cd rossmann_forecasting
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run notebooks sequentially:
   - `01_EDA_FeatureEngineering.ipynb`
   - `02_StatisticalModels_ARIMA_Prophet.ipynb`
   - `03_MLModels_XGBoost_RF.ipynb`
   - `04_CausalImpact_Analysis.ipynb`
   - `05_Explainability_SHAP_PDP.ipynb`

✅ All modular scripts (`src/`) are available if you want to pipeline this project.

---

## 🔮 Future Work

- Automate hyperparameter optimization (Optuna).
- Build live forecasting dashboard using Streamlit.
- Expand causal analysis to include external events (e.g., holidays, weather).
- Model multiple stores using hierarchical time series (HTS).

---

## 💜 License

This project is licensed under the **MIT License** — free to use and modify.

---


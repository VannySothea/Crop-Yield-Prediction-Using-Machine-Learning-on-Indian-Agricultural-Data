# 🌾 Crop Yield Prediction Using Machine Learning

## 📘 Project Overview

This project aims to predict the yield of agricultural crops in India using historical crop production data and machine learning. By analyzing features such as crop type, state, season, area, and production, the model estimates the expected yield per unit area. This prediction can assist farmers, policymakers, and agricultural planners in making informed decisions.

---

## 📚 Table of Contents

- [🌍 Societal and Industrial Impact](#-societal-and-industrial-impact)
- [🎯 Objectives](#-objectives)
- [🔍 Research Questions](#-research-questions)
- [🧠 Contributions](#-contributions)
- [📦 Dataset](#-dataset)
- [🛠️ Methodology](#-methodology)
- [🤖 ML Models Used](#-ml-models-used)
- [❓ Why These Models?](#-why-these-models)
- [📏 Evaluation Metrics](#-evaluation-metrics)
- [⚙️ Hyperparameter Tuning](#️-hyperparameter-tuning)

---

## 🌍 Societal and Industrial Impact

- 💡 Helps farmers optimize crop selection and land use.
- 📈 Supports government planning for food security and distribution.
- 🚜 Enables better management of agricultural resources.
- 🌱 Contributes to sustainable farming practices.

---

## 🎯 Objectives

- Understand how crop yield varies by location, season, and crop type.
- Train ML models to predict yield based on structured input data.
- Evaluate and compare model performance using regression metrics.
- Provide insights that can improve agricultural productivity and planning.

---

## 🔍 Research Questions

- **What** factors influence agricultural crop yields across regions and crops in India?
- **Why** is it important to predict crop yields accurately?
- **How** can machine learning models be trained on historical data to estimate yields effectively?

---

## 🧠 Contributions

- Built a machine learning pipeline to predict yield based on features like crop type, location, season, and area
- Identified key features impacting yield
- Compared regression models for performance on real-world data
- Proposed insights for agricultural planning and decision-making

---

## 📦 Dataset

- **Type:** Secondary Dataset  
- **Source:** [Kaggle - India Agriculture Crop Production](https://www.kaggle.com/datasets/pyatakov/india-agriculture-crop-production/data)  
- **Columns:**
  - `State`
  - `District`
  - `Crop`
  - `Year`
  - `Season`
  - `Area` (in hectares)
  - `Production` (in tonnes)
  - `Yield` = `Production / Area`

---

## 🛠️ Methodology

1. Data cleaning and preprocessing
2. Feature encoding (categorical to numerical)
3. Train-test split
4. Model training (Regression)
5. Evaluation and tuning
6. Result interpretation and visualization

---

## 🤖 ML Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

---

## ❓ Why These Models?

- Well-suited for tabular regression tasks
- Easy to interpret
- Handle both numerical and encoded categorical data
- Tree-based models capture nonlinear relationships

---

## 📏 Evaluation Metrics

- R² Score (Coefficient of Determination)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

---

## ⚙️ Hyperparameter Tuning

Used GridSearchCV / manual tuning for:
- `n_estimators`, `max_depth` (Random Forest)
- `max_depth`, `min_samples_split` (Decision Tree)


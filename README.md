# 🌾 Optimizing Crop Selection Using Machine Learning for Sustainable Agriculture

## 📘 Project Overview

This project can significantly improve crop yield and farming efficiency, particularly in resource-constrained environments. It supports sustainable agricultural practices by recommending the best crops based on environmental and soil parameters, reducing input waste, and increasing food security.

---

## 📚 Table of Contents

- [🌍 Societal and Industrial Impact](#-societal-and-industrial-impact)
- [🎯 Problem Statement](#-problem-statement)
- [🔍 Research Questions](#-research-questions)
- [🧠 Contributions](#-contributions)
- [📦 Dataset](#-dataset)
- [🛠️ Methodology](#-methodology)
- [🤖 ML Models Used](#-ml-models-used)
- [❓ Why These Models?](#-why-these-models)
- [📏 Evaluation Metrics](#-evaluation-metrics)
- [⚙️ Hyperparameter Tuning](#️-hyperparameter-tuning)
- [⚙️ Reflection and Argument](#reflection-and-argument)

---

## 🌍 Societal and Industrial Impact

This project can significantly improve crop yield and farming efficiency, particularly in resource-constrained environments. It supports sustainable agricultural practices by recommending the best crops based on environmental and soil parameters, reducing input waste, and increasing food security.

---

## 🎯 Problem Statement

Farmers often lack access to scientific tools that can help them decide which crops to grow under given environmental and soil conditions. This leads to inefficient use of resources and lower yields.

---

## 🔍 Research Questions

- **What** What is the best crop to grow given a specific combination of N, P, K, temperature, humidity, pH, and rainfall?
- **Why** To maximize agricultural productivity and ensure optimal use of natural resources.
- **How** By applying machine learning classification models to predict the most suitable crop based on environmental and soil features.

---

## 🧠 Contributions

- Developed a machine learning model that accurately predicts optimal crop types.
- Preprocessed and analyzed a real-world agricultural dataset.
- Provided a practical decision-making tool for farmers and agronomists.
- Evaluated multiple ML models and selected the best-performing one based on accuracy and efficiency.

---

## 📦 Dataset

- **Type:** Secondary Dataset: [agricultural_production_optimization.csv ](https://github.com/VannySothea/Optimizing-Crop-Selection-Using-Machine-Learning-for-Sustainable-Agriculture/blob/main/agricultural_production_optimization.csv) 
- **The dataset includes the following features:**
  - `N` (Nitrogen content in the soil)
  - `P` (Phosphorus content)
  - `K` (Potassium content)
  - `Temperature` (In Celsius)
  - `Humidity` (In percentage)
  - `pH` (pH value of the soil)
  - `Rainfall` (In mm)
  - `Label` (Crop name - target variable)

---

## 🛠 Methodology

1. Data Preprocessing (cleaning, encoding, normalization)
2. Exploratory Data Analysis (EDA)
3. Feature Selection
4. Model Selection
5. Model Training and Testing
6. Evaluation and Fine-tuning
7. Visualization of results

---

## 🤖 ML Models Used

Random Forest Classifier, K-Nearest Neighbors, Support Vector Machine, Decision Tree

---

## ❓ Why These Models?

Random Forest was chosen due to its robustness, ability to handle feature importance, and high accuracy in classification tasks with limited data. It also handles overfitting better than a single decision tree.

---

## 📏 Evaluation Metrics

1. Accuracy
2. Confusion Matrix
3. Classification Report (Precision, Recall, F1-score)
4. Cross-validation (e.g., 5-fold)

---

## ⚙️ Hyperparameter Tuning

For Random Forest:
n_estimators = 100
max_depth = 10
random_state = 42

---

## Reflection and Argument

This project demonstrates how AI can be used in agriculture to improve decision-making and productivity. The predictive model offers a data-driven approach to crop planning. Future improvements could include integrating real-time weather data and farmer feedback for even more adaptive models.

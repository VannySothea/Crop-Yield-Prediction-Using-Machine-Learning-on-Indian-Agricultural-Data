# 🌾 Optimizing Crop Selection Using Machine Learning for Sustainable Agriculture

## 📘 Project Overview

This project leverages machine learning to recommend the most suitable crop for a given set of soil and environmental conditions. The system promotes sustainable agriculture by enhancing crop yields, reducing input waste, and improving decision-making for farmers. It is particularly useful in regions where access to agronomic expertise and resources is limited.

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
- [🔍 Findings from Clustering](#-findings-from-clustering)
- [🧠 Reflection and Argument](#-reflection-and-argument)

---

## 🌍 Societal and Industrial Impact

- Provides a data-driven solution for farmers to choose the right crop based on soil and climate.
- Helps in resource optimization: water, fertilizers, and land can be used more efficiently.
- Aids in climate-resilient agriculture, important for regions vulnerable to environmental changes.
- Can be used by agricultural departments or agri-tech companies to build advisory systems.

---

## 🎯 Problem Statement

Farmers often make crop selection decisions based on tradition or intuition, which can lead to poor yields. There is a need for a scientific, automated tool to determine the most appropriate crop based on real-time soil and environmental data.

---

## 🔍 Research Questions

- **What** is the most suitable crop for a given combination of soil and climatic parameters?
- **Why** is it important to guide crop selection using data?
- **How** can machine learning models improve crop selection accuracy and sustainability?

---

## 🧠 Contributions

- Built an interactive EDA tool to summarize and compare crop requirements.
- Used K-Means clustering to identify natural groupings in crop types.
- Trained a Logistic Regression classifier to predict the optimal crop for a given set of conditions.
- Visualized crop classification performance using a confusion matrix.
- Extracted insights about seasonal crops and specific nutrient requirements.

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

1. Data Loading & Cleaning:
   - Checked for null values and ensured data integrity.
3. Exploratory Data Analysis (EDA):
   - Used interactive widgets to explore crop-specific requirements for nutrients, temperature, humidity, etc.
   - Compared average requirements across crops.
5. Clustering with K-Means:
   - Grouped crops into clusters based on environmental and soil needs.
7. Supervised Learning:
   - Applied Logistic Regression for crop prediction.
9. Model Evaluation:
    - Calculated accuracy, precision, recall, F1-score.
    - Visualized performance using a confusion matrix.

---

## 🤖 ML Models Used

- K-Means Clustering: for unsupervised pattern discovery among crops.
- Logistic Regression: for supervised crop prediction.

---

## ❓ Why These Models?

- K-Means provides useful insights into natural groupings of crop types, beneficial for segmentation.
- Logistic Regression offers simplicity, interpretability, and reasonable performance for multiclass classification problems.

---

## 📏 Evaluation Metrics

For model evaluation, the following metrics were used:
1. **Accuracy**: Overall prediction correctness.
2. **Precision (Weighted)**: How precise each prediction is across all classes.
3. **Recall (Weighted)**: Ability of the model to capture all relevant crops.
4. **F1 Score (Weighted)**: Harmonic mean of precision and recall.
5. **Confusion Matrix**: Visual tool to identify misclassified crops.

**This Model**
```
Accuracy: 0.9681818181818181
Precision: 0.9699452867394045
Recall: 0.9681818181818181
F1 Score: 0.9681168080082031
```

---

## ⚙️ Hyperparameter Tuning

Basic parameters for KMeans:
n_clusters = 4
init = 'k-means++'
max_iter = 300
random_state = 0

Logistic Regression was used with default settings for initial evaluation. Further improvements could include:
- GridSearchCV for hyperparameter tuning
- Trying more advanced classifiers (e.g., Random Forest, XGBoost)

---

## 🔍 Findings from Clustering

Crops were grouped into four natural clusters based on their resource requirements.
Each cluster revealed a different environmental preference, helping identify patterns for:
- Low-Nutrient Crops
- High-Rainfall Crops
- High-Temperature Crops
- Balanced Crops

---

## 🧠 Reflection and Argument

**Strengths:**
- Clear methodology with interactive analysis
- Practical real-world application for farmers
- Easy-to-understand model (Logistic Regression)

**Limitations:**
- Logistic Regression may underperform with complex, non-linear data
- Dataset may not account for regional crop constraints or pests

**Future Work:**
- Integrate satellite and real-time IoT sensor data
- Incorporate regional constraints and climate anomalies
- Extend to multi-crop recommendation and yield prediction

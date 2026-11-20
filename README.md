<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12&height=140&section=header&text=Customer%20Churn%20Prediction&fontSize=30&fontColor=ffffff&animation=fadeIn&fontAlignY=50" width="100%"/>

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-2196F3?style=for-the-badge&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

</div>

## 📌 Overview

End-to-end machine learning pipeline for predicting customer churn using **XGBoost, Random Forest, and Logistic Regression** with SHAP-based explainability. Identifies key drivers of customer attrition to enable proactive retention strategies.

## 📊 Model Performance

| Metric | Score |
|:---|:---|
| **Accuracy** | 87% |
| **ROC-AUC** | 0.89 |
| **Best Model** | XGBoost |

## 🔑 Key Features

- **Multi-model comparison**: XGBoost vs Random Forest vs Logistic Regression
- **SHAP explainability**: Feature importance and individual prediction explanations
- **End-to-end pipeline**: Data cleaning → Feature engineering → Modelling → Evaluation
- **Class imbalance handling**: SMOTE / class weight balancing
- **Visualizations**: ROC curves, confusion matrix, SHAP summary plots

## 📦 Dependencies

```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
imbalanced-learn
jupyter
```

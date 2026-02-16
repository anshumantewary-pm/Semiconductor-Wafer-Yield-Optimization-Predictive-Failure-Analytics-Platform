#Semiconductor Wafer Defect Prediction and Yield Analysis (SECOM Dataset)

Overview
This project builds a semiconductor manufacturing analytics system using the SECOM dataset. The goal is to predict wafer defects using high-dimensional sensor data and translate predictive insights into meaningful business impact.

The system focuses on structured numeric data rather than image-based defect detection. It demonstrates how machine learning, statistical validation, and financial modeling can work together to improve manufacturing decision-making.

A detailed PDF explaining the full system architecture, methodology, modeling approach, and financial simulation is included in this repository.

#What This Project Does

This project addresses three key objectives:
Predict wafer defects using 590 anonymized sensor measurements.
Identify the most influential sensor features contributing to defect outcomes.
Convert predicted defect probabilities into financial impact estimates.
Instead of stopping at model accuracy, the system evaluates how predictive improvements can reduce production losses and improve yield.

#Dataset

This project uses the SECOM Manufacturing Dataset, originally published through the UCI Machine Learning Repository and available on Kaggle.

#Dataset characteristics:
1,567 observations
590 sensor features
Binary pass/fail label
Each row represents a manufactured unit, and each column represents a sensor measurement recorded during production.

#System Components

The project includes the following components:
Data cleaning and preprocessing
Feature engineering and selection
Predictive modeling (Logistic Regression, XGBoost, LSTM)
Model evaluation and validation
Explainable AI using SHAP
Statistical validation techniques
Financial impact simulation

#Project Structure

data/ – Dataset files
notebooks/ – Exploratory and modeling notebooks
docs/ – System documentation and design files
diagrams/ – Architecture and ER diagrams
financial_model/ – ROI and cost simulation models

#Documentation

A comprehensive PDF is included in this repository explaining:
System architecture
Data processing pipeline
Modeling methodology
Evaluation metrics
Financial simulation framework
Business implications

This document provides a structured explanation of the entire system.

#Technologies Used

SQL
Python (Pandas, NumPy, Scikit-learn, XGBoost, SHAP)
R (Statistical validation)
Microsoft Excel (Financial modeling)

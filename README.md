# Customer Churn Prediction and Analysis

## Overview
This project aims to predict customer churn (i.e., whether a customer will leave the service) using machine learning models. The dataset contains demographic information, service details, and customer behavior. The goal is to build a model that can accurately predict churn and present the results in an interactive way using Streamlit.

## Features
- **Data Loading:** Reading and cleaning the Telco Customer Churn dataset.
- **Data Preprocessing:**
  - Handling missing values and converting data types (e.g., converting `TotalCharges` to numeric).
  - Normalizing features for model training.
- **Exploratory Data Analysis (EDA):**
  - Visualizing distributions of key features like tenure, monthly charges, and churn rates.
  - Analyzing relationships between features and their impact on churn.
- **Feature Engineering:** Creating relevant features to improve model accuracy.
- **Model Training and Evaluation:**
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - Hyperparameter tuning using GridSearchCV
- **Evaluation Metrics:** Confusion matrix, accuracy, and classification reports.
- **Streamlit Application:** An interactive dashboard to visualize customer data and churn predictions, allowing users to explore model predictions and churn insights.

## Libraries Used
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- Streamlit

## Dataset
The dataset used is the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data) dataset from Kaggle. It includes information like:
- `customerID`: Unique ID for each customer.
- Demographic data such as `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
- Service-related features such as `tenure`, `PhoneService`, `InternetService`.
- Payment data like `MonthlyCharges` and `TotalCharges`.
- `Churn`: Target variable indicating whether the customer has churned.

Ensure the dataset is available in the working directory.

## Project Structure
- **`Customer_Churn_Prediction_and_Analysis.ipynb`:** Jupyter Notebook containing the full project workflow.
- **`app.py`:** Streamlit application file for deploying the interactive dashboard.
- **`data/`:** Directory containing the Telco Customer Churn dataset.
- **`images/`:** Directory for saving visualizations.
- **`requirements.txt`:** List of dependencies required to run the project.

## How to Run
1. Clone the repository:
   ```bash
   git clone <https://github.com/saifisvibinn/Customer-Churn-Prediciton>
   ```
2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Customer_Churn_Prediction_and_Analysis.ipynb
   ```
3. Launch the Streamlit app:
   ```bash
   streamlit run Customer_Churn_Prediction.py
   ```

## Results
- Achieved an accuracy of **82%** using the optimized Logistic Regression model.
- The best parameters from `GridSearchCV` were:
  - **C:** 100
  - **Penalty:** l1 (Lasso regularization)
  - **Solver:** liblinear
- Key factors influencing customer churn include contract type, monthly charges, and tenure.
- The Streamlit app provides a user-friendly interface to explore predictions and insights interactively.

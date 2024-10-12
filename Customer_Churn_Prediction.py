

import streamlit as st
import numpy as np
import mlflow.sklearn
import pandas as pd

# Load the model from MLflow (replace with the correct model version)
model = mlflow.sklearn.load_model(model_uri="models:/Customer_Churn_Prediction/1")

# Streamlit app
st.title("Customer Churn Prediction")

# Collect customer details from user input
st.write("Please input the customer's information:")

# Input fields matching the features used in your model
tenure = st.number_input('Tenure (in months)', min_value=0, max_value=72, value=1)
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=0.0)
total_charges = st.number_input('Total Charges', min_value=0.0, value=0.0)
senior_citizen = st.selectbox('Is the customer a senior citizen?', ['No', 'Yes'])
contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
gender = st.selectbox('Gender', ['Male', 'Female'])
dependents = st.selectbox('Has dependents?', ['No', 'Yes'])
partner = st.selectbox('Has a partner?', ['No', 'Yes'])
phone_service = st.selectbox('Has phone service?', ['No', 'Yes'])
multiple_lines = st.selectbox('Multiple lines?', ['No', 'Yes', 'No phone service'])
internet_service = st.selectbox('Internet service type?', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Has online security?', ['No', 'Yes', 'No internet service'])
online_backup = st.selectbox('Has online backup?', ['No', 'Yes', 'No internet service'])
device_protection = st.selectbox('Has device protection?', ['No', 'Yes', 'No internet service'])
tech_support = st.selectbox('Has tech support?', ['No', 'Yes', 'No internet service'])
streaming_tv = st.selectbox('Has streaming TV?', ['No', 'Yes', 'No internet service'])
streaming_movies = st.selectbox('Has streaming movies?', ['No', 'Yes', 'No internet service'])
paperless_billing = st.selectbox('Uses paperless billing?', ['No', 'Yes'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# One-hot encoding for categorical features (like contract, payment method, etc.)
contract_onehot = pd.get_dummies(contract, prefix='Contract')
payment_method_onehot = pd.get_dummies(payment_method, prefix='PaymentMethod')

# Map categorical variables (binary mappings)
senior_citizen = 1 if senior_citizen == 'Yes' else 0
gender = 1 if gender == 'Male' else 0
partner = 1 if partner == 'Yes' else 0
dependents = 1 if dependents == 'Yes' else 0
paperless_billing = 1 if paperless_billing == 'Yes' else 0

# Convert other Yes/No inputs into numeric (1/0)
features_binary = {
    "PhoneService": 1 if phone_service == 'Yes' else 0,
    "MultipleLines": 1 if multiple_lines == 'Yes' else 0,
    "InternetService": 1 if internet_service != 'No' else 0,  # Add this line
    "OnlineSecurity": 1 if online_security == 'Yes' else 0,
    "OnlineBackup": 1 if online_backup == 'Yes' else 0,
    "DeviceProtection": 1 if device_protection == 'Yes' else 0,
    "TechSupport": 1 if tech_support == 'Yes' else 0,
    "StreamingTV": 1 if streaming_tv == 'Yes' else 0,
    "StreamingMovies": 1 if streaming_movies == 'Yes' else 0
}

# Prepare the feature array (this should match the feature order used during training)
features = []

# Numeric features
features.extend([tenure, monthly_charges, total_charges])

# Binary features
binary_features = [
    senior_citizen, 
    gender, 
    partner, 
    dependents,
    features_binary['PhoneService'],
    features_binary['MultipleLines'],
    features_binary['InternetService'],
    features_binary['OnlineSecurity'], 
    features_binary['OnlineBackup'],
    features_binary['DeviceProtection'], 
    features_binary['TechSupport'],
    features_binary['StreamingTV'], 
    features_binary['StreamingMovies'],
    paperless_billing
]
features.extend(binary_features)

# Categorical features (one-hot encoded)
# Internet Service (3 features)
features.extend([
    1 if internet_service == 'DSL' else 0,
    1 if internet_service == 'Fiber optic' else 0,
    1 if internet_service == 'No' else 0
])

# Contract (3 features)
features.extend([
    1 if contract == 'Month-to-month' else 0,
    1 if contract == 'One year' else 0,
    1 if contract == 'Two year' else 0
])

# Payment Method (4 features)
features.extend([
    1 if payment_method == 'Electronic check' else 0,
    1 if payment_method == 'Mailed check' else 0,
    1 if payment_method == 'Bank transfer (automatic)' else 0,
    1 if payment_method == 'Credit card (automatic)' else 0
])

# Multiple Lines (3 features)
features.extend([
    1 if multiple_lines == 'No' else 0,
    1 if multiple_lines == 'Yes' else 0,
    1 if multiple_lines == 'No phone service' else 0
])

# Convert to numpy array
features = np.array(features).reshape(1, -1)

# Print the number of features for debugging
print(f"Number of features: {features.shape[1]}")

# Ensure the number of features matches your model's input
if st.button('Predict'):
    try:
        # Make the prediction
        prediction = model.predict(features)
        
        # Display the prediction result
        if prediction[0] == 1:
            st.write("The customer is likely to churn.")
        else:
            st.write("The customer is not likely to churn.")
    except Exception as e:
        st.write(f"Error: {e}")
        st.write(f"Number of features: {features.shape[1]}")


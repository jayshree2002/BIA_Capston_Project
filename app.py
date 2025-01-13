import streamlit as st
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the trained models and scaler
kmeans_model = joblib.load("kmeans_model.pkl")
pca_model = joblib.load("pca_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the cluster labels (you can modify based on your analysis)
cluster_labels = {
    0: "High Spenders",
    1: "Low Frequency Shoppers",
    2: "Occasional Shoppers",
    3: "Frequent Bargain Shoppers"
}

# Original feature names
feature_names = [
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT", "TENURE"
]

# Streamlit app layout
st.title("Customer Segmentation")

st.write("Enter customer data (17 features) below to see which cluster they belong to:")

# Create a dynamic input form based on feature names
input_features = []
for feature_name in feature_names:
    input_value = st.number_input(f"{feature_name}", min_value=0.0, max_value=100.0, step=0.1)
    input_features.append(input_value)

# Convert the input list to a numpy array
input_features = np.array(input_features).reshape(1, -1)

# Ensure the scaler is using the correct number of features
scaled_input = scaler.transform(input_features)

# Apply PCA to the scaled input
pca_input = pca_model.transform(scaled_input)

# Predict the cluster label using KMeans
predicted_label = kmeans_model.predict(pca_input)

# Display the result
st.write(f"The customer belongs to : **{cluster_labels[predicted_label[0]]}**")

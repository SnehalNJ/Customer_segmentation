import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
model_path = "final_model.pkl"
scaler_path = "scaler.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Customer Segmentation App")
st.write("Enter customer details to determine their segment.")

# Define user inputs
feature_names = ["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"]  # Adjust as per actual features
user_inputs = []

for feature in feature_names:
    value = st.number_input(f"Enter {feature}", min_value=0.0, max_value=100000.0, value=0.0, step=0.01)
    user_inputs.append(value)

# Convert inputs to numpy array
input_array = np.array(user_inputs).reshape(1, -1)

# Scale the inputs
scaled_input = scaler.transform(input_array)

# Predict segment
if st.button("Predict Segment"):
    segment = model.predict(scaled_input)[0]
    st.success(f"Predicted Customer Segment: {segment}")

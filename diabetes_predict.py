import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model_path = 'XGB.pkl'  # Update with your model path
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Initialize scalers with mean and std from training data
scaler_age = StandardScaler()
scaler_bmi = StandardScaler()
scaler_HbA1c = StandardScaler()
scaler_blood_glucose = StandardScaler()

# Precomputed mean and std for scaling (use the values you have from training data)
scaler_age.mean_ = np.array([41.794326])
scaler_age.scale_ = np.array([22.462948])
scaler_bmi.mean_ = np.array([27.321461])
scaler_bmi.scale_ = np.array([6.767716])
scaler_HbA1c.mean_ = np.array([5.532609])
scaler_HbA1c.scale_ = np.array([1.073232])
scaler_blood_glucose.mean_ = np.array([138.218231])
scaler_blood_glucose.scale_ = np.array([40.909771])

# Set up the Streamlit app
st.title("Diabetes Prediction App")

# Collect user input
gender = st.selectbox("Gender", options=["Male", "Female", "Others"])
age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
hypertension = st.selectbox("Hypertension", options=["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", options=["No", "Yes"])
smoking_history = st.selectbox("Smoking History", options=["never", "No Info", "former", "current", "not current", "ever"])
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
hba1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0, max_value=300.0, value=120.0, step=0.1)

# Convert categorical inputs to numerical values
gender_map = {"Male": 1, "Female": 0, "Others": 2}
hypertension_map = {"No": 0, "Yes": 1}
heart_disease_map = {"No": 0, "Yes": 1}
smoking_history_map = {"never": 0, "No Info": 1, "former": 2, "current": 3, "not current": 4, "ever": 5}

gender = gender_map[gender]
hypertension = hypertension_map[hypertension]
heart_disease = heart_disease_map[heart_disease]
smoking_history = smoking_history_map[smoking_history]

# Arrange input data for prediction
input_data = np.array([[gender, hypertension, heart_disease, smoking_history, age, bmi, hba1c_level, blood_glucose_level]])

# Scale the numerical inputs
input_data[:, 4] = scaler_age.transform(input_data[:, [4]].reshape(-1, 1)).flatten()  # Scale age
input_data[:, 5] = scaler_bmi.transform(input_data[:, [5]].reshape(-1, 1)).flatten()  # Scale BMI
input_data[:, 6] = scaler_HbA1c.transform(input_data[:, [6]].reshape(-1, 1)).flatten()  # Scale HbA1c
input_data[:, 7] = scaler_blood_glucose.transform(input_data[:, [7]].reshape(-1, 1)).flatten()  # Scale blood glucose


# Add a prediction button
if st.button('Predict'):
    # Make the prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0]

    # Display the result
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.write("Oops!! You have diabetes.")
    else:
        st.write("Congrats! You don't have diabetes.")


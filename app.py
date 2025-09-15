import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the models and scalers
def load_model_and_scaler(disease_type):
    """Loads the pre-trained model and scaler for a given disease."""
    try:
        model = joblib.load(f'{disease_type}_model.pkl')
        scaler = joblib.load(f'{disease_type}_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Could not find model or scaler files for {disease_type}. Please ensure {disease_type}_model.pkl and {disease_type}_scaler.pkl are in the same directory.")
        return None, None

# Define a generic prediction function
def predict_disease(disease_type, input_data, model, scaler):
    """Makes a prediction for a given disease using the loaded model and scaler."""
    if model and scaler:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[:, 1][0]
        return prediction, probability
    return None, None

# Page Configuration
st.set_page_config(
    page_title="Disease Prediction App",
    page_icon="⚕️",
    layout="wide"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Prediction Model", ["Heart Disease", "Breast Cancer", "Diabetes"])

# --- Heart Disease Prediction Page ---
if page == "Heart Disease":
    st.title("❤️ Heart Disease Prediction")
    st.markdown("Enter the patient's data to predict the likelihood of heart disease.")

    # Load model and scaler
    heart_model, heart_scaler = load_model_and_scaler('heart')

    if heart_model and heart_scaler:
        # Input features based on data_prep_heart.ipynb
        with st.form("heart_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=1, max_value=120, value=50)
                sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
                cp = st.selectbox("Chest Pain Type (cp)", options=[(1, "Typical Angina"), (2, "Atypical Angina"), (3, "Non-Anginal Pain"), (4, "Asymptomatic")])
                trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
                chol = st.number_input("Serum Cholestoral (chol)", min_value=100, max_value=600, value=200)
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)])
                restecg = st.selectbox("Resting Electrocardiographic Results", options=[(0, "Normal"), (1, "Having ST-T wave abnormality"), (2, "Probable or definite left ventricular hypertrophy")])

            with col2:
                thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
                exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)])
                oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
                slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[(1, "Upsloping"), (2, "Flat"), (3, "Downsloping")])
                ca = st.number_input("Number of Major Vessels (0-3) Colored by Fluoroscopy (ca)", min_value=0, max_value=3, value=0)
                thal = st.selectbox("Thal", options=[(3, "Normal"), (6, "Fixed Defect"), (7, "Reversable Defect")])

            submitted = st.form_submit_button("Predict Heart Disease")

        if submitted:
            # Prepare input for prediction
            input_list = [age, sex[1], cp[0], trestbps, chol, fbs[1], restecg[0], thalach, exang[1], oldpeak, slope[0], ca, thal[0]]
            heart_input = np.array([input_list])

            # Make prediction
            prediction, probability = predict_disease('heart', heart_input, heart_model, heart_scaler)
            
            if prediction is not None:
                st.subheader("Prediction Result")
                if prediction[0] == 1:
                    st.error(f"Prediction: Heart Disease Detected 💔")
                    st.write(f"Probability of having heart disease: {probability * 100:.2f}%")
                else:
                    st.success("Prediction: No Heart Disease Detected ✅")
                    st.write(f"Probability of having heart disease: {probability * 100:.2f}%")

# --- Breast Cancer Prediction Page ---
elif page == "Breast Cancer":
    st.title("🌸 Breast Cancer Prediction")
    st.markdown("Enter the tumor features to predict whether the tumor is malignant (0) or benign (1).")

    # Load model and scaler
    breast_model, breast_scaler = load_model_and_scaler('breast')

    if breast_model and breast_scaler:
        # Input features based on data_prep_breastCancer.ipynb
        features = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
            'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
            'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
            'area error', 'smoothness error', 'compactness error', 'concavity error',
            'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area',
            'worst smoothness', 'worst compactness', 'worst concavity',
            'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]

        with st.form("breast_form"):
            input_dict = {}
            cols = st.columns(3)
            for i, feature in enumerate(features):
                input_dict[feature] = cols[i % 3].number_input(f"{feature.replace('_', ' ').title()}", value=0.0)

            submitted = st.form_submit_button("Predict Breast Cancer")

        if submitted:
            # Prepare input for prediction
            input_list = [input_dict[f] for f in features]
            breast_input = np.array([input_list])

            # Make prediction
            prediction, probability = predict_disease('breast', breast_input, breast_model, breast_scaler)

            if prediction is not None:
                st.subheader("Prediction Result")
                if prediction[0] == 0:
                    st.error("Prediction: Malignant Tumor Detected 📛")
                    st.write(f"Probability of being malignant: {probability * 100:.2f}%")
                else:
                    st.success("Prediction: Benign Tumor Detected ✅")
                    st.write(f"Probability of being malignant: {probability * 100:.2f}%")

# --- Diabetes Prediction Page ---
elif page == "Diabetes":
    st.title("🩸 Diabetes Prediction")
    st.markdown("Enter the patient's data to predict the likelihood of diabetes.")

    # Load model and scaler
    diabetes_model, diabetes_scaler = load_model_and_scaler('diabetes')

    if diabetes_model and diabetes_scaler:
        # Input features based on data_prep_diabetes.ipynb
        with st.form("diabetes_form"):
            col1, col2 = st.columns(2)
            with col1:
                pregnancies = st.number_input("Pregnancies", min_value=0, value=6)
                glucose = st.number_input("Glucose", min_value=0, value=148)
                blood_pressure = st.number_input("Blood Pressure", min_value=0, value=72)
                skin_thickness = st.number_input("Skin Thickness", min_value=0, value=35)
            with col2:
                insulin = st.number_input("Insulin", min_value=0, value=0)
                bmi = st.number_input("BMI", min_value=0.0, value=33.6)
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.627)
                age = st.number_input("Age", min_value=1, value=50)

            submitted = st.form_submit_button("Predict Diabetes")

        if submitted:
            # Prepare input for prediction
            input_list = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
            diabetes_input = np.array([input_list])

            # Make prediction
            prediction, probability = predict_disease('diabetes', diabetes_input, diabetes_model, diabetes_scaler)

            if prediction is not None:
                st.subheader("Prediction Result")
                if prediction[0] == 1:
                    st.error("Prediction: Diabetes Detected 🚨")
                    st.write(f"Probability of having diabetes: {probability * 100:.2f}%")
                else:
                    st.success("Prediction: No Diabetes Detected ✅")
                    st.write(f"Probability of having diabetes: {probability * 100:.2f}%")

# Main page content
if st.session_state.get('initial_page', True):
    st.title("Welcome to the Disease Prediction App 🩺")
    st.markdown("""
    This application uses machine learning models to predict the likelihood of three different diseases:
    - **Heart Disease**
    - **Breast Cancer**
    - **Diabetes**

    To get started, please select a disease from the sidebar to the left and enter the patient's data.
    """)
    st.session_state['initial_page'] = False
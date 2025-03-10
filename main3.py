import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Multi-Disease Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Cache model loading to improve performance
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# Define available models
models = {
    "Diabetes Prediction": "models/diabetes_model.pkl",
    "Heart Disease Prediction": "models/heart_model.pkl",
    "Lung Cancer Prediction": "models/LungCancer.pkl"
}

# App header with styling
st.markdown("""
    <h1 style='text-align: center;'>Multi-Disease Prediction System</h1>
    <p style='text-align: center;'>An AI-powered tool to predict various medical conditions</p>
    <hr>
""", unsafe_allow_html=True)

# Sidebar for model selection
with st.sidebar:
    st.header("Select Disease")
    selected_model_name = st.selectbox("Choose prediction model:", list(models.keys()))
    st.info(f"Model selected: {selected_model_name}")
    
    # Load model based on selection
    try:
        model = load_model(models[selected_model_name])
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Main content area
def diabetes_prediction_ui():
    """UI for diabetes prediction inputs and prediction"""
    st.header("Diabetes Prediction")
    st.markdown("Enter patient information to predict diabetes risk")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=80)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=85)
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=50.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, 
                             help="A function that scores likelihood of diabetes based on family history")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
    
    # Prediction button with styling
    if st.button("Predict Diabetes Risk", type="primary"):
        with st.spinner("Processing..."):
            user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                  insulin, bmi, dpf, age]])
            prediction = model.predict(user_data)
            probability = model.predict_proba(user_data)
            
            st.subheader("Prediction Result:")
            if prediction[0] == 1:
                st.error(f"⚠️ Patient likely has diabetes (Confidence: {probability[0][1]:.2%})")
                st.markdown("""
                    **Recommendation:** Please consult with a healthcare professional for further evaluation.
                """)
            else:
                st.success(f"✅ Patient unlikely to have diabetes (Confidence: {probability[0][0]:.2%})")
                st.markdown("""
                    **Recommendation:** Maintain a healthy lifestyle and continue regular check-ups.
                """)

def heart_disease_prediction_ui():
    """UI for heart disease prediction inputs and prediction"""
    st.header("Heart Disease Prediction")
    st.markdown("Enter patient information to predict heart disease risk")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
        gender = st.radio("Gender", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", 
                                 ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    
    with col2:
        cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=600, value=200)
        fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])
        resting_ecg = st.selectbox("Resting ECG Results", 
                                  ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        max_heart_rate = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
    
    # Convert categorical features to numerical
    gender_encoded = 1 if gender == "Male" else 0
    fasting_bs_encoded = 1 if fasting_bs == "Yes" else 0
    
    chest_pain_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    chest_pain_encoded = chest_pain_mapping[chest_pain]
    
    ecg_mapping = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    resting_ecg_encoded = ecg_mapping[resting_ecg]
    
    # For simplicity, we'll use a subset of features (adjust according to your model)
    if st.button("Predict Heart Disease Risk", type="primary"):
        with st.spinner("Processing..."):
            features = np.array([[age, gender_encoded, chest_pain_encoded, resting_bp, 
                                 cholesterol, fasting_bs_encoded, resting_ecg_encoded, max_heart_rate]])
            
            # Assuming these are the features your model expects
            try:
                prediction = model.predict(features)
                
                st.subheader("Prediction Result:")
                if prediction[0] == 1:
                    st.error("⚠️ Patient likely has heart disease")
                    st.markdown("""
                        **Recommendation:** Consult with a cardiologist immediately for comprehensive evaluation.
                    """)
                else:
                    st.success("✅ Patient unlikely to have heart disease")
                    st.markdown("""
                        **Recommendation:** Continue heart-healthy habits and regular check-ups.
                    """)
            except Exception as e:
                st.error(f"Prediction error: {e}. Your heart disease model may require different features.")

def lung_cancer_prediction_ui():
    """UI for lung cancer prediction inputs and prediction"""
    st.header("Lung Cancer Prediction")
    st.markdown("Enter patient information to predict lung cancer risk")
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age (years)", 20, 100, 50)
        gender = st.radio("Gender", ["Male", "Female"])
        smoking = st.radio("Smoking Status", ["Yes", "No"])
        yellow_fingers = st.radio("Yellow Fingers", ["Yes", "No"])
        anxiety = st.radio("Anxiety", ["Yes", "No"])
    
    with col2:
        chronic_disease = st.radio("Chronic Disease", ["Yes", "No"])
        fatigue = st.radio("Fatigue", ["Yes", "No"])
        allergy = st.radio("Allergy", ["Yes", "No"])
        wheezing = st.radio("Wheezing", ["Yes", "No"])
    
    with col3:
        coughing = st.radio("Coughing", ["Yes", "No"])
        shortness_of_breath = st.radio("Shortness of Breath", ["Yes", "No"])
        swallowing_difficulty = st.radio("Swallowing Difficulty", ["Yes", "No"])
        chest_pain = st.radio("Chest Pain", ["Yes", "No"])
    
    # Convert Yes/No inputs to 1/0
    features = {
        'Age': age,
        'Gender': 1 if gender == "Male" else 0,
        'Smoking': 1 if smoking == "Yes" else 0,
        'Yellow_Fingers': 1 if yellow_fingers == "Yes" else 0,
        'Anxiety': 1 if anxiety == "Yes" else 0,
        'Chronic Disease': 1 if chronic_disease == "Yes" else 0,
        'Fatigue': 1 if fatigue == "Yes" else 0,
        'Allergy': 1 if allergy == "Yes" else 0,
        'Wheezing': 1 if wheezing == "Yes" else 0,
        'Coughing': 1 if coughing == "Yes" else 0,
        'Shortness of Breath': 1 if shortness_of_breath == "Yes" else 0,
        'Swallowing Difficulty': 1 if swallowing_difficulty == "Yes" else 0,
        'Chest Pain': 1 if chest_pain == "Yes" else 0
    }
    
    # Convert to DataFrame to match model's expected input format
    input_df = pd.DataFrame([features])
    
    # Prediction button with styling
    if st.button("Predict Lung Cancer Risk", type="primary"):
        with st.spinner("Processing..."):
            try:
                prediction = model.predict(input_df)
                
                st.subheader("Prediction Result:")
                if prediction[0] == 1:
                    st.error("⚠️ Patient shows signs consistent with lung cancer risk")
                    st.markdown("""
                        **Recommendation:** Urgent consultation with a pulmonologist is advised.
                        Further diagnostic tests like CT scan may be needed.
                    """)
                else:
                    st.success("✅ Patient unlikely to have lung cancer")
                    st.markdown("""
                        **Recommendation:** Continue monitoring for symptoms and 
                        maintain regular health check-ups.
                    """)
            except Exception as e:
                st.error(f"Prediction error: {e}. Please check the feature format required by your model.")

# Display the appropriate UI based on selected model
if selected_model_name == "Diabetes Prediction":
    diabetes_prediction_ui()
elif selected_model_name == "Heart Disease Prediction":
    heart_disease_prediction_ui()
elif selected_model_name == "Lung Cancer Prediction":
    lung_cancer_prediction_ui()

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 40px;'>
    <p><em>Disclaimer: This application is for educational purposes only. 
    Always consult with healthcare professionals for medical advice.</em></p>
</div>
""", unsafe_allow_html=True)
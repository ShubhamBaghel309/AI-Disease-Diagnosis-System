import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define model paths
MODELS_DIR = "models"
DIABETES_MODEL_PATH = os.path.join(MODELS_DIR, "diabetes_model.pkl")
HEART_MODEL_PATH = os.path.join(MODELS_DIR, "heart_model.pkl")
LUNG_CANCER_MODEL_PATH = os.path.join(MODELS_DIR, "LungCancer.pkl")

# Cache resource loading
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {str(e)}")
        return None

# Load all models at startup
def load_all_models():
    models = {
        "diabetes": load_model(DIABETES_MODEL_PATH),
        "heart": load_model(HEART_MODEL_PATH),
        "lung": load_model(LUNG_CANCER_MODEL_PATH)
    }
    return models

# Load models
models = load_all_models()

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.75rem;
        color: #2874A6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.35rem;
        color: #3498DB;
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #C8E6C9;  /* Darker shade of green */
        border-left: 5px solid #2E7D32;  /* Dark green border */
        color: #1B5E20;  /* Very dark green text for contrast */
    }
    .warning-box {
        background-color: #FFE0B2;  /* Darker shade of orange */
        border-left: 5px solid #E65100;  /* Dark orange border */
        color: #BF360C;  /* Very dark orange text for contrast */
    }
    .danger-box {
        background-color: #FFCDD2;  /* Darker shade of red */
        border-left: 5px solid #C62828;  /* Dark red border */
        color: #B71C1C;  /* Very dark red text for contrast */
    }
    .feature-box {
        background-color: #E3F2FD;  /* Light blue background */
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #90CAF9;  /* Add border for better definition */
    }
    .stButton>button {
        width: 100%;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #37474F;  /* Darker text for better readability */
        text-align: center;
        padding: 1rem;
        background-color: #ECEFF1;  /* Slightly darker background */
        border-radius: 0.5rem;
        margin-top: 2rem;
        border: 1px solid #CFD8DC;  /* Add subtle border */
    }
    /* New class for prediction result headers with better contrast */
    .prediction-header {
        color: #212121;  /* Near black for maximum contrast */
        font-weight: 600;  /* Semi-bold for better visibility */
    }
    /* Improve recommendation text visibility */
    .recommendation {
        font-weight: 500;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">Multi-Disease Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem;">An AI-powered tool for early disease risk assessment</p>', unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/medical-healthcare-blue-color_1017-26807.jpg", width=250)
    st.markdown("## Navigation")
    disease_option = st.radio(
        "Select Disease to Predict",
        ["Home", "Diabetes Prediction", "Heart Disease Prediction", "Lung Cancer Prediction"]
    )
    
    st.markdown("---")
    st.markdown("## About")
    st.info("""
    This application uses machine learning to predict the risk of various diseases based on patient health metrics.
    
    All predictions are probabilistic and should be used for informational purposes only.
    """)

# Home page
def show_home():
    st.markdown('<h2 class="sub-header">Welcome to the Disease Prediction System</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This application uses machine learning algorithms to predict the risk of multiple diseases:
        
        - **Diabetes**: Based on glucose levels, BMI, age, and other factors
        - **Heart Disease**: Based on cardiac symptoms, cholesterol levels, and patient history
        - **Lung Cancer**: Based on symptoms and risk factors
        
        ### How to Use
        1. Select a disease from the sidebar
        2. Enter the required medical information
        3. Click the "Predict" button to see results
        4. Consult with healthcare professionals for proper diagnosis
        
        ### Importance of Early Detection
        Early detection of diseases significantly increases treatment success rates. This tool aims to help identify potential risks that warrant further medical investigation.
        """)
    
    with col2:
        st.image("https://img.freepik.com/free-vector/doctor-examining-patient-clinic-illustrated_23-2148856559.jpg", width=300)
        
    # Model status
    st.markdown('<h3 class="section-header">Model Status</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if models["diabetes"] is not None:
            st.success("✅ Diabetes Model: Loaded")
        else:
            st.error("❌ Diabetes Model: Not Found")
    
    with col2:
        if models["heart"] is not None:
            st.success("✅ Heart Disease Model: Loaded")
        else:
            st.error("❌ Heart Disease Model: Not Found")
    
    with col3:
        if models["lung"] is not None:
            st.success("✅ Lung Cancer Model: Loaded")
        else:
            st.error("❌ Lung Cancer Model: Not Found")

# Diabetes prediction page
def show_diabetes_prediction():
    st.markdown('<h2 class="sub-header">Diabetes Risk Prediction</h2>', unsafe_allow_html=True)
    
    # Check if model is loaded
    if models["diabetes"] is None:
        st.error("Diabetes model not found. Please ensure the model file exists in the models directory.")
        return
    
    st.write("Enter the following health metrics to predict diabetes risk:")
    
    with st.container():
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=17, value=1)
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
            
        with col2:
            insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=79)
            bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=60.0, value=25.0, format="%.1f")
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
            age = st.number_input("Age (years)", min_value=18, max_value=120, value=33)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button
    if st.button("Predict Diabetes Risk", type="primary"):
        try:
            # Create input data for prediction
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            
            # Make prediction
            prediction = models["diabetes"].predict(input_data)
            probability = models["diabetes"].predict_proba(input_data)[0]
            
            # Display result with improved contrast
            st.markdown("### Prediction Result")
            
            if prediction[0] == 1:
                risk_probability = probability[1] * 100
                risk_level = "High" if risk_probability > 70 else "Moderate"
                
                st.markdown(f"""
                <div class="prediction-box danger-box">
                    <h3 class="prediction-header">⚠️ {risk_level} Risk of Diabetes</h3>
                    <p>The model predicts a {risk_probability:.1f}% probability of diabetes.</p>
                    <p class="recommendation"><strong>Recommendation:</strong> Please consult with a healthcare professional for a comprehensive evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                safe_probability = probability[0] * 100
                st.markdown(f"""
                <div class="prediction-box success-box">
                    <h3 class="prediction-header">✅ Low Risk of Diabetes</h3>
                    <p>The model predicts a {safe_probability:.1f}% probability of not having diabetes.</p>
                    <p class="recommendation"><strong>Recommendation:</strong> Maintain a healthy lifestyle with regular exercise and balanced diet.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display reference ranges
            with st.expander("Reference Ranges & Information"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **Glucose Levels:**
                    - Normal: <100 mg/dL
                    - Prediabetes: 100-125 mg/dL
                    - Diabetes: ≥126 mg/dL
                    
                    **BMI:**
                    - Underweight: <18.5
                    - Normal weight: 18.5-24.9
                    - Overweight: 25-29.9
                    - Obesity: ≥30
                    """)
                
                with col2:
                    st.markdown("""
                    **Blood Pressure:**
                    - Normal: <120/80 mm Hg
                    - Elevated: 120-129/<80 mm Hg
                    - Hypertension Stage 1: 130-139/80-89 mm Hg
                    - Hypertension Stage 2: ≥140/≥90 mm Hg
                    
                    **Insulin (Fasting):**
                    - Normal: <25 mU/L
                    """)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please ensure all input values are within normal ranges.")

# Heart disease prediction page
def show_heart_disease_prediction():
    st.markdown('<h2 class="sub-header">Heart Disease Risk Prediction</h2>', unsafe_allow_html=True)
    
    # Check if model is loaded
    if models["heart"] is None:
        st.error("Heart disease model not found. Please ensure the model file exists in the models directory.")
        return
    
    st.write("Enter cardiac health information to predict heart disease risk:")
    
    with st.container():
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
            sex = st.radio("Sex", ["Male", "Female"])
            chest_pain = st.selectbox("Chest Pain Type", 
                ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=130)
            cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=230)
        
        with col2:
            fbs = st.radio("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
            rest_ecg = st.selectbox("Resting ECG", [
                "Normal", 
                "ST-T Wave Abnormality",
                "Left Ventricular Hypertrophy"
            ])
            max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
            exercise_angina = st.radio("Exercise-Induced Angina", ["No", "Yes"])
            st_depression = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, value=0.0, step=0.1)
        
        with col3:
            st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
            num_vessels = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
            thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Encode categorical features
    sex_encoded = 1 if sex == "Male" else 0
    fbs_encoded = 1 if fbs == "Yes" else 0
    exercise_angina_encoded = 1 if exercise_angina == "Yes" else 0
    
    cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    cp_encoded = cp_mapping[chest_pain]
    
    rest_ecg_mapping = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    rest_ecg_encoded = rest_ecg_mapping[rest_ecg]
    
    slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope_encoded = slope_mapping[st_slope]
    
    thal_mapping = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
    thal_encoded = thal_mapping[thalassemia]
    
    # Prediction button
    if st.button("Predict Heart Disease Risk", type="primary"):
        try:
            # Create input data - format depends on how your model was trained
            # You may need to adjust these features based on your model's expectations
            input_data = np.array([[
                age, sex_encoded, cp_encoded, resting_bp, cholesterol, fbs_encoded,
                rest_ecg_encoded, max_hr, exercise_angina_encoded, st_depression,
                slope_encoded, num_vessels, thal_encoded
            ]])
            
            # Make prediction
            prediction = models["heart"].predict(input_data)
            
            # Get probability if available
            try:
                probability = models["heart"].predict_proba(input_data)[0]
                has_probability = True
            except:
                has_probability = False
            
            # Display result with improved contrast
            st.markdown("### Prediction Result")
            
            if prediction[0] == 1:
                if has_probability:
                    risk_probability = probability[1] * 100
                    risk_text = f"with {risk_probability:.1f}% probability"
                else:
                    risk_text = "with high confidence"
                    
                st.markdown(f"""
                <div class="prediction-box danger-box">
                    <h3 class="prediction-header">⚠️ High Risk of Heart Disease</h3>
                    <p>The model predicts heart disease {risk_text}.</p>
                    <p class="recommendation"><strong>Recommendation:</strong> Please consult with a cardiologist for a thorough evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                if has_probability:
                    safe_probability = probability[0] * 100
                    safe_text = f"with {safe_probability:.1f}% probability"
                else:
                    safe_text = "with high confidence"
                    
                st.markdown(f"""
                <div class="prediction-box success-box">
                    <h3 class="prediction-header">✅ Low Risk of Heart Disease</h3>
                    <p>The model predicts no heart disease {safe_text}.</p>
                    <p class="recommendation"><strong>Recommendation:</strong> Continue heart-healthy lifestyle habits.</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Additional information
            with st.expander("Heart Health Information"):
                st.markdown("""
                ### Risk Factors for Heart Disease
                
                - High blood pressure
                - High cholesterol
                - Diabetes
                - Obesity and being overweight
                - Smoking
                - Family history of heart disease
                - Lack of physical activity
                - High stress levels
                
                ### Prevention Tips
                
                - Eat a heart-healthy diet
                - Exercise regularly
                - Maintain a healthy weight
                - Don't smoke or use tobacco
                - Limit alcohol use
                - Manage stress
                - Regular health screenings
                """)
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please ensure all input values are within expected ranges.")

# Lung cancer prediction page
def show_lung_cancer_prediction():
    st.markdown('<h2 class="sub-header">Lung Cancer Risk Prediction</h2>', unsafe_allow_html=True)
    
    # Check if model is loaded
    if models["lung"] is None:
        st.error("Lung cancer model not found. Please ensure the model file exists in the models directory.")
        return
    
    # Add debug option to show column names
    debug_mode = st.checkbox("Debug mode (show model column names)", value=False)
    if debug_mode:
        try:
            # Try to access feature names from the model if available
            feature_names = models["lung"].feature_names_in_ if hasattr(models["lung"], 'feature_names_in_') else None
            if feature_names is not None:
                st.info("Expected feature names from model:")
                st.code(feature_names.tolist())
            else:
                st.info("Could not retrieve feature names directly from model")
                # Show the exact expected feature names based on error messages
                st.info("Using hardcoded expected column names:")
                expected_cols = ['Unnamed: 0', 'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 
                               'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 
                               'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
                               'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
                st.code(expected_cols)
        except Exception as e:
            st.error(f"Error accessing model feature names: {e}")
    
    st.write("Answer these questions to assess lung cancer risk:")
    
    with st.container():
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", min_value=20, max_value=90, value=45)
            gender = st.radio("Gender", ["Male", "Female"])
            smoking = st.radio("Do you smoke?", ["Yes", "No"])
            yellow_fingers = st.radio("Yellow fingers?", ["Yes", "No"])
            anxiety = st.radio("Do you have anxiety?", ["Yes", "No"])
            peer_pressure = st.radio("Do you experience peer pressure?", ["Yes", "No"])
        
        with col2:
            chronic_disease = st.radio("Do you have chronic diseases?", ["Yes", "No"])
            fatigue = st.radio("Do you experience fatigue?", ["Yes", "No"])
            allergy = st.radio("Do you have allergies?", ["Yes", "No"])
            wheezing = st.radio("Do you experience wheezing?", ["Yes", "No"])
            alcohol_consuming = st.radio("Do you consume alcohol?", ["Yes", "No"])
        
        with col3:
            coughing = st.radio("Do you have a persistent cough?", ["Yes", "No"])
            shortness_of_breath = st.radio("Do you experience shortness of breath?", ["Yes", "No"])
            swallowing_difficulty = st.radio("Do you have difficulty swallowing?", ["Yes", "No"])
            chest_pain = st.radio("Do you experience chest pain?", ["Yes", "No"])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction function with exact column names
    if st.button("Predict Lung Cancer Risk", type="primary"):
        try:
            # Create input data with EXACT column names from model
            # Note: 'FATIGUE ' and 'ALLERGY ' have trailing spaces
            input_data = {
                'Unnamed: 0': 0,
                'GENDER': 1 if gender == "Male" else 0,
                'AGE': age,
                'SMOKING': 1 if smoking == "Yes" else 0,
                'YELLOW_FINGERS': 1 if yellow_fingers == "Yes" else 0,
                'ANXIETY': 1 if anxiety == "Yes" else 0,
                'PEER_PRESSURE': 1 if peer_pressure == "Yes" else 0,
                'CHRONIC DISEASE': 1 if chronic_disease == "Yes" else 0,
                'FATIGUE ': 1 if fatigue == "Yes" else 0,  # Note the trailing space
                'ALLERGY ': 1 if allergy == "Yes" else 0,  # Note the trailing space
                'WHEEZING': 1 if wheezing == "Yes" else 0,
                'ALCOHOL CONSUMING': 1 if alcohol_consuming == "Yes" else 0,
                'COUGHING': 1 if coughing == "Yes" else 0,
                'SHORTNESS OF BREATH': 1 if shortness_of_breath == "Yes" else 0,
                'SWALLOWING DIFFICULTY': 1 if swallowing_difficulty == "Yes" else 0,
                'CHEST PAIN': 1 if chest_pain == "Yes" else 0
            }
            
            # Create DataFrame with exactly matching columns
            input_df = pd.DataFrame([input_data])
            
            if debug_mode:
                st.write("Input data columns:", input_df.columns.tolist())
            
            # Make prediction
            prediction = models["lung"].predict(input_df)
            
            # Get probability if available
            try:
                probability = models["lung"].predict_proba(input_df)[0]
                has_probability = True
            except Exception as e:
                if debug_mode:
                    st.error(f"Could not get probability: {e}")
                probability = [0.5, 0.5]  # Default if not available
                has_probability = False
            
            # Display result with improved contrast
            st.markdown("### Prediction Result")
            
            if prediction[0] == 1:
                prob_text = f"{probability[1]:.1%} probability" if has_probability else "high likelihood"
                
                st.markdown(f"""
                <div class="prediction-box danger-box">
                    <h3 class="prediction-header">⚠️ High Risk of Lung Cancer</h3>
                    <p>The model predicts {prob_text} of lung cancer.</p>
                    <p class="recommendation"><strong>Recommendation:</strong> Please consult with a healthcare professional immediately for proper evaluation and testing.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show key risk factors
                risk_factors = []
                if smoking == "Yes": risk_factors.append("Smoking")
                if yellow_fingers == "Yes": risk_factors.append("Yellow Fingers")
                if anxiety == "Yes": risk_factors.append("Anxiety")
                if peer_pressure == "Yes": risk_factors.append("Peer Pressure")
                if chronic_disease == "Yes": risk_factors.append("Chronic Disease")
                if fatigue == "Yes": risk_factors.append("Fatigue")
                if allergy == "Yes": risk_factors.append("Allergy")
                if wheezing == "Yes": risk_factors.append("Wheezing")
                if alcohol_consuming == "Yes": risk_factors.append("Alcohol Consumption")
                if coughing == "Yes": risk_factors.append("Coughing")
                if shortness_of_breath == "Yes": risk_factors.append("Shortness of Breath")
                if swallowing_difficulty == "Yes": risk_factors.append("Swallowing Difficulty")
                if chest_pain == "Yes": risk_factors.append("Chest Pain")
                
                if risk_factors:
                    st.markdown("#### Key Risk Factors Identified:")
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
                        
            else:
                prob_text = f"{probability[0]:.1%} probability" if has_probability else "high likelihood"
                
                st.markdown(f"""
                <div class="prediction-box success-box">
                    <h3 class="prediction-header">✅ Low Risk of Lung Cancer</h3>
                    <p>The model predicts {prob_text} of not having lung cancer.</p>
                    <p class="recommendation"><strong>Recommendation:</strong> Continue healthy habits and regular check-ups.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional information
            with st.expander("Lung Cancer Information"):
                st.markdown("""
                ### Key Risk Factors for Lung Cancer
                
                - Smoking: The #1 risk factor, responsible for 85% of lung cancer cases
                - Exposure to secondhand smoke
                - Exposure to radon gas
                - Exposure to asbestos and other carcinogens
                - Family history of lung cancer
                - Previous radiation therapy to the chest
                
                ### Warning Signs of Lung Cancer
                
                - Persistent cough
                - Chest pain
                - Hoarseness
                - Weight loss
                - Bloody or rust-colored sputum
                - Shortness of breath
                - Recurring infections such as bronchitis and pneumonia
                - New onset of wheezing
                """)
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            
            if debug_mode:
                with st.expander("Error Details"):
                    st.code(str(e))
                    # Extract column information from error if possible
                    error_str = str(e)
                    if "feature names" in error_str.lower():
                        lines = error_str.split('\n')
                        st.write("Error contains feature name information. Please check column names carefully.")
                        for line in lines:
                            if "feature names" in line.lower():
                                st.code(line)
            
            st.warning("""
            **There was an error with the lung cancer prediction model.**
            
            This could be due to a column name mismatch between the model's expected input and what was provided.
            
            Try enabling debug mode to see more information about the expected column names.
            """)

# Display the appropriate page based on sidebar selection
if disease_option == "Home":
    show_home()
elif disease_option == "Diabetes Prediction":
    show_diabetes_prediction()
elif disease_option == "Heart Disease Prediction":
    show_heart_disease_prediction()
elif disease_option == "Lung Cancer Prediction":
    show_lung_cancer_prediction()

# Footer with disclaimer
st.markdown("""
<div class="disclaimer">
    <p><strong>Disclaimer:</strong> This application is designed for educational and informational purposes only. 
    It is not intended to be a substitute for professional medical advice, diagnosis, or treatment.
    Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
</div>
""", unsafe_allow_html=True)


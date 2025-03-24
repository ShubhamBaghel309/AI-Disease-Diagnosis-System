# Health Guardian AI - Multi-Disease Prediction System

A machine learning-powered web application for early disease risk detection and assessment with an enhanced, modern user interface.

![Application Screenshot](https://img.freepik.com/free-photo/medical-banner-with-doctor-working-hospital_23-2149611193.jpg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Disease Models](#disease-models)
- [Technical Architecture](#technical-architecture)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [UI Enhancements](#ui-enhancements)
- [Model Performance](#model-performance)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

## Overview

Health Guardian AI is an advanced AI-powered tool designed to predict the risk of various diseases based on patient health metrics, symptoms, and laboratory values. The application features a modern, intuitive user interface and uses trained machine learning models to provide probabilistic risk assessments for multiple conditions including diabetes, heart disease, Parkinson's disease, lung cancer, and thyroid disorders.

This project demonstrates the practical application of machine learning in healthcare for early disease detection and risk assessment, enabling timely intervention and potentially improving patient outcomes.

## Features

- **Enhanced, modern UI**: Clean, attractive design with medical-themed styling
- **Multiple disease prediction**: Single platform for assessing five different health conditions
- **Interactive risk visualization**: Dynamic risk factor display with progress bars and tagging
- **Educational components**: Expandable sections with medical reference information
- **Real-time feedback**: Instant predictions with visual results and recommendations
- **Input validation**: Guided numerical inputs with appropriate ranges
- **Responsive design**: Optimized layout that works across various screen sizes
- **Visual category indicators**: BMI categories, risk levels, and other medical indicators
- **Loading animations**: Visual feedback during prediction processing
- **Debug options**: Technical tools for troubleshooting model issues

## Disease Models

### Diabetes Prediction
Predicts diabetes risk based on key health indicators including:
- Number of pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- BMI (Body Mass Index)
- Diabetes Pedigree Function (genetic predisposition)
- Age

### Heart Disease Prediction
Assesses heart disease risk using cardiac health parameters such as:
- Age and gender
- Chest pain characteristics
- Resting blood pressure
- Cholesterol levels
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate
- Exercise-induced angina
- ST depression and slope
- Number of major vessels
- Thalassemia type

### Parkinson's Disease Prediction
Evaluates Parkinson's disease risk based on voice analysis metrics:
- Vocal frequency measurements (Fo, Fhi, Flo)
- Jitter measurements (variations in frequency)
- Shimmer measurements (variations in amplitude)
- Noise-to-harmonics ratios
- Nonlinear measurements (RPDE, DFA, PPE)

### Lung Cancer Prediction
Evaluates lung cancer risk based on symptoms and risk factors including:
- Smoking history
- Physical symptoms (coughing, chest pain, etc.)
- Environmental and lifestyle factors
- Pre-existing conditions
- Age and gender

### Thyroid Function Assessment
Analyzes thyroid hormone levels to predict hypothyroidism:
- TSH levels
- T3 and T4 measurements
- Patient demographics
- Current medications

## Technical Architecture

### Framework
- **Streamlit**: Front-end web application framework
- **Python**: Core programming language

### Machine Learning Components
- **scikit-learn**: Model training and evaluation
- **pickle**: Model serialization and persistence
- **pandas/numpy**: Data handling and numerical operations

### Project Structure
```
HealthGuardianAI/
│
├── app.py                # Enhanced Streamlit application with modern UI
├── main.py               # Alternative implementation with different styling
├── main3.py              # Alternative implementation
├── main4.py              # Original version of the app
├── fix_diabetes_model.py # Utility script for diabetes model
├── fix_lung_model.py     # Utility script for lung cancer model
├── test_lung_model.py    # Test script for lung cancer model
├── train_lungcancer.py   # Training script for lung cancer model
├── Diabeties.ipynb       # Notebook for diabetes model development
├── images/               # Directory for images and icons
├── Models/               # Directory for trained models
│   ├── Diabetes.sav
│   ├── heart_disease_model.sav
│   ├── parkinsons_model.sav
│   ├── lungs_disease_model.sav
│   └── Thyroid_model.sav
└── README.md             # Project documentation
```

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```
   git clone <repository-url>
   cd HealthGuardianAI
   ```

2. Install required dependencies:
   ```
   pip install streamlit pandas numpy scikit-learn streamlit-option-menu pillow
   ```

3. Run the enhanced application:
   ```
   streamlit run app.py
   ```
   
   Or run the alternative version:
   ```
   streamlit run main.py
   ```

4. Access the application:
   Open your web browser and navigate to `http://localhost:8501`

### Model Files
Ensure the following trained model files are in the Models directory:
- `Models/Diabetes.sav`
- `Models/heart_disease_model.sav`
- `Models/parkinsons_model.sav`
- `Models/lungs_disease_model.sav`
- `Models/Thyroid_model.sav`

## Usage Guide

### Disease Selection
1. Choose from five different disease prediction options
2. Select either using the enhanced visual cards or the dropdown menu
3. View disease-specific input forms and educational information

### Diabetes Prediction
1. Enter patient metrics including glucose levels, BMI, age, etc.
2. View real-time BMI category assessment
3. Submit for prediction results and risk factor analysis
4. Access educational content about diabetes through the expandable section

### Heart Disease Prediction
1. Input cardiovascular parameters including chest pain characteristics
2. Review real-time cardiovascular risk assessment visualization
3. Submit for prediction with personalized recommendations
4. Access heart health information through the expandable section

### Parkinson's Disease Prediction
1. Enter voice analysis parameters organized by measure type
2. Submit for prediction with detailed results
3. Review educational content about voice analysis in Parkinson's detection

### Lung Cancer Prediction
1. Input patient information and risk factors
2. Indicate symptom presence using Yes/No selections
3. View dynamic risk assessment score and visualization
4. Enable debug mode for technical insights if needed
5. Access lung cancer information through the expandable section

### Thyroid Function Assessment
1. Enter laboratory values including TSH, T3, and T4 levels
2. Review TSH level assessment visualization
3. Submit for prediction with identified risk indicators
4. Access thyroid function educational information

## UI Enhancements

The application features a significantly improved user interface with:

- **Modern medical theme**: Professional color scheme with blue medical accents
- **Animated loading**: Visual feedback during prediction calculations
- **Risk visualization**: Dynamic progress bars and tagged risk factors
- **Enhanced typography**: Improved readability with appropriate text sizing
- **Card-based layout**: Clear visual organization of elements
- **Interactive elements**: Hover effects and visual feedback
- **Tooltips**: Contextual help for input parameters
- **Responsive containers**: Adaptable layouts for different screen sizes
- **Gradient styling**: Smooth color transitions for visual appeal
- **Medical icons**: Distinct icons for each disease type
- **Result highlight boxes**: Clear visual distinction between positive/negative results

## Model Performance

### Diabetes Model
- **Algorithm**: Logistic Regression
- **Accuracy**: ~75% on validation data
- **Key predictors**: Glucose level, BMI, Age

### Heart Disease Model
- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~80% on validation data
- **Key predictors**: Chest pain type, ST depression, Number of vessels

### Parkinson's Disease Model
- **Algorithm**: XGBoost Classifier
- **Accuracy**: ~85% on validation data
- **Key predictors**: Various voice parameter measurements

### Lung Cancer Model
- **Algorithm**: Support Vector Machine
- **Accuracy**: ~85% on validation data
- **Key predictors**: Smoking, Age, Respiratory symptoms

### Thyroid Model
- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~80% on validation data
- **Key predictors**: TSH level, T4 level

## Future Enhancements

- **User accounts**: Save and track predictions over time
- **PDF reports**: Generate downloadable health assessment reports
- **Mobile optimization**: Dedicated mobile app version
- **API integration**: Connect with medical record systems
- **Additional disease models**: Expand to include more conditions
- **Multilingual support**: Interface in multiple languages
- **Advanced visualizations**: Interactive charts for risk factor relationships
- **Telemedicine integration**: Connect with healthcare providers
- **Wearable device integration**: Import data from fitness trackers and medical devices

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

Health Guardian AI is designed for educational and informational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers for any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read or seen in this application.

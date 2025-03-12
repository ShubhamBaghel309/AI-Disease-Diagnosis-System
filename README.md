# Multi-Disease Prediction System

A machine learning-powered web application for early disease risk detection and assessment.

![Application Screenshot](https://img.freepik.com/free-vector/medical-healthcare-blue-color_1017-26807.jpg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Disease Models](#disease-models)
- [Technical Architecture](#technical-architecture)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [Development Process](#development-process)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Multi-Disease Prediction System is an AI-powered tool designed to predict the risk of various diseases based on patient health metrics and symptoms. The application uses machine learning models trained on medical datasets to provide probabilistic risk assessments for diabetes, heart disease, and lung cancer.

This project demonstrates the practical application of machine learning in healthcare for early disease detection and risk assessment, potentially enabling timely intervention and improved patient outcomes.

## Features

- **User-friendly interface**: Clean, intuitive design optimized for healthcare professionals and individuals
- **Multiple disease prediction**: Single platform for assessing multiple health conditions
- **Evidence-based predictions**: Machine learning models trained on established medical datasets
- **Real-time feedback**: Instant predictions with confidence levels and recommendations
- **Educational information**: Reference ranges and health information for better context
- **Accessibility features**: High-contrast design for improved readability
- **Responsive layout**: Works across desktop and tablet devices
- **Model diagnostics**: Debug mode for technical users to validate model behavior

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

### Lung Cancer Prediction
Evaluates lung cancer risk based on symptoms and risk factors including:
- Smoking history
- Physical symptoms (coughing, chest pain, etc.)
- Environmental and genetic factors
- Lifestyle indicators
- Pre-existing conditions

## Technical Architecture

### Framework
- **Streamlit**: Front-end web application framework
- **Python**: Core programming language

### Machine Learning Components
- **scikit-learn**: Model training and evaluation
- **joblib/pickle**: Model serialization and persistence
- **pandas/numpy**: Data handling and numerical operations

### Project Structure
```
Multi-Disease-Prediction/
│
├── main4.py               # Main Streamlit application
├── main3.py               # Alternative Streamlit application
├── fix_diabetes_model.py  # Utility script for diabetes model
├── Diabeties.ipynb        # Notebook for diabetes model development
├── models/                # Directory for trained models
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   └── LungCancer.pkl
├── diabetes_data.csv      # Dataset for diabetes model
└── README.md              # Project documentation
```

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```
   git clone <repository-url>
   cd Multi-Disease-Prediction
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run main4.py
   ```

4. Access the application:
   Open your web browser and navigate to `http://localhost:8501`

### Model Files
Place the following trained model files in the appropriate directories:
- `models/diabetes_model.pkl`
- `models/heart_model.pkl`
- `models/LungCancer.pkl`

## Usage Guide

### Home Page
- View the application overview and model status
- Access educational information about disease prediction

### Diabetes Prediction
1. Navigate to "Diabetes Prediction" from the sidebar
2. Enter patient health metrics including glucose, BMI, age, etc.
3. Click "Predict Diabetes Risk" button
4. View the prediction result, confidence level, and recommendations
5. Access reference ranges for better interpretation

### Heart Disease Prediction
1. Navigate to "Heart Disease Prediction" from the sidebar
2. Enter cardiac health parameters
3. Click "Predict Heart Disease Risk" button
4. Review prediction results and suggested next steps

### Lung Cancer Prediction
1. Navigate to "Lung Cancer Prediction" from the sidebar
2. Answer questions about symptoms and risk factors
3. Click "Predict Lung Cancer Risk" button
4. Review results, identified risk factors, and recommendations

## Model Performance

### Diabetes Model
- **Algorithm**: Logistic Regression
- **Accuracy**: ~75% on validation data
- **Key predictors**: Glucose level, BMI, Age

### Heart Disease Model
- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~80% on validation data
- **Key predictors**: Chest pain type, ST depression, Number of vessels

### Lung Cancer Model
- **Algorithm**: Gradient Boosting Classifier
- **Accuracy**: ~85% on validation data
- **Key predictors**: Smoking, Age, Coughing, Shortness of breath

## Development Process

1. **Data Collection**: Gathered relevant medical datasets for each disease
2. **Exploratory Data Analysis**: Analyzed features and distributions
3. **Data Preprocessing**: Cleaned data and handled missing values
4. **Feature Engineering**: Created additional relevant features
5. **Model Selection**: Evaluated multiple algorithms for each disease
6. **Hyperparameter Tuning**: Optimized model parameters
7. **Model Evaluation**: Validated using cross-validation techniques
8. **Model Deployment**: Integrated with Streamlit front-end
9. **User Interface Development**: Created intuitive interface
10. **Testing & Validation**: Verified predictions against expected outcomes

## Future Enhancements

- **Additional Disease Models**: Expand to include kidney disease, stroke risk assessment
- **Patient History Tracking**: Enable saving and tracking predictions over time
- **Advanced Visualizations**: Add interactive charts for risk factor relationships
- **API Integration**: Connect with medical record systems
- **Mobile Application**: Develop companion mobile app for on-the-go assessments
- **Multi-language Support**: Add internationalization for global access
- **Explainable AI**: Improve transparency of prediction rationale
- **User Authentication**: Add secure login for healthcare professionals

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

This application is designed for educational and informational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

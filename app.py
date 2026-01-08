from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models and scaler if they exist
try:
    with open('logistic_regression_model.pkl', 'rb') as f:
        logreg = pickle.load(f)
    with open('xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    models_loaded = True
except FileNotFoundError:
    models_loaded = False

@app.route('/')
@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/predict_form')
def predict_form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return render_template('form.html', prediction="Models not loaded. Please train and save models first.")

    # Get form data
    senior_citizen = int(request.form['SeniorCitizen'])
    tenure = float(request.form['tenure'])
    monthly_charges = float(request.form['MonthlyCharges'])
    total_charges = float(request.form['TotalCharges'])
    gender = request.form['gender']
    partner = request.form['Partner']
    dependents = request.form['Dependents']
    phone_service = request.form['PhoneService']
    multiple_lines = request.form['MultipleLines']
    internet_service = request.form['InternetService']
    online_security = request.form['OnlineSecurity']
    online_backup = request.form['OnlineBackup']
    device_protection = request.form['DeviceProtection']
    tech_support = request.form['TechSupport']
    streaming_tv = request.form['StreamingTV']
    streaming_movies = request.form['StreamingMovies']
    contract = request.form['Contract']
    paperless_billing = request.form['PaperlessBilling']
    payment_method = request.form['PaymentMethod']

    # Create DataFrame
    input_data = pd.DataFrame({
        'SeniorCitizen': [senior_citizen],
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'gender_Male': [1 if gender == 'Male' else 0],
        'Partner_Yes': [1 if partner == 'Yes' else 0],
        'Dependents_Yes': [1 if dependents == 'Yes' else 0],
        'PhoneService_Yes': [1 if phone_service == 'Yes' else 0],
        'MultipleLines_Yes': [1 if multiple_lines == 'Yes' else 0],
        'InternetService_Fiber optic': [1 if internet_service == 'Fiber optic' else 0],
        'InternetService_No': [1 if internet_service == 'No' else 0],
        'OnlineSecurity_Yes': [1 if online_security == 'Yes' else 0],
        'OnlineBackup_Yes': [1 if online_backup == 'Yes' else 0],
        'DeviceProtection_Yes': [1 if device_protection == 'Yes' else 0],
        'TechSupport_Yes': [1 if tech_support == 'Yes' else 0],
        'StreamingTV_Yes': [1 if streaming_tv == 'Yes' else 0],
        'StreamingMovies_Yes': [1 if streaming_movies == 'Yes' else 0],
        'Contract_One year': [1 if contract == 'One year' else 0],
        'Contract_Two year': [1 if contract == 'Two year' else 0],
        'PaperlessBilling_Yes': [1 if paperless_billing == 'Yes' else 0],
        'PaymentMethod_Credit card (automatic)': [1 if payment_method == 'Credit card (automatic)' else 0],
        'PaymentMethod_Electronic check': [1 if payment_method == 'Electronic check' else 0],
        'PaymentMethod_Mailed check': [1 if payment_method == 'Mailed check' else 0]
    })

    # Scale numerical features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    # Predict
    prediction = logreg.predict(input_data)[0]
    probability = logreg.predict_proba(input_data)[0][1]

    result = "Churn" if prediction == 1 else "No Churn"

    prediction_text = f"<strong>Prediction:</strong> {result}<br><strong>Probability of Churn:</strong> {probability:.2f}"

    return render_template('form.html', prediction=prediction_text)

@app.route('/predict_xgb', methods=['POST'])
def predict_xgb():
    if not models_loaded:
        return render_template('form.html', prediction_xgb="Models not loaded. Please train and save models first.")

    # Get form data
    senior_citizen = int(request.form['SeniorCitizen'])
    tenure = float(request.form['tenure'])
    monthly_charges = float(request.form['MonthlyCharges'])
    total_charges = float(request.form['TotalCharges'])
    gender = request.form['gender']
    partner = request.form['Partner']
    dependents = request.form['Dependents']
    phone_service = request.form['PhoneService']
    multiple_lines = request.form['MultipleLines']
    internet_service = request.form['InternetService']
    online_security = request.form['OnlineSecurity']
    online_backup = request.form['OnlineBackup']
    device_protection = request.form['DeviceProtection']
    tech_support = request.form['TechSupport']
    streaming_tv = request.form['StreamingTV']
    streaming_movies = request.form['StreamingMovies']
    contract = request.form['Contract']
    paperless_billing = request.form['PaperlessBilling']
    payment_method = request.form['PaymentMethod']

    # Create DataFrame
    input_data = pd.DataFrame({
        'SeniorCitizen': [senior_citizen],
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'gender_Male': [1 if gender == 'Male' else 0],
        'Partner_Yes': [1 if partner == 'Yes' else 0],
        'Dependents_Yes': [1 if dependents == 'Yes' else 0],
        'PhoneService_Yes': [1 if phone_service == 'Yes' else 0],
        'MultipleLines_Yes': [1 if multiple_lines == 'Yes' else 0],
        'InternetService_Fiber optic': [1 if internet_service == 'Fiber optic' else 0],
        'InternetService_No': [1 if internet_service == 'No' else 0],
        'OnlineSecurity_Yes': [1 if online_security == 'Yes' else 0],
        'OnlineBackup_Yes': [1 if online_backup == 'Yes' else 0],
        'DeviceProtection_Yes': [1 if device_protection == 'Yes' else 0],
        'TechSupport_Yes': [1 if tech_support == 'Yes' else 0],
        'StreamingTV_Yes': [1 if streaming_tv == 'Yes' else 0],
        'StreamingMovies_Yes': [1 if streaming_movies == 'Yes' else 0],
        'Contract_One year': [1 if contract == 'One year' else 0],
        'Contract_Two year': [1 if contract == 'Two year' else 0],
        'PaperlessBilling_Yes': [1 if paperless_billing == 'Yes' else 0],
        'PaymentMethod_Credit card (automatic)': [1 if payment_method == 'Credit card (automatic)' else 0],
        'PaymentMethod_Electronic check': [1 if payment_method == 'Electronic check' else 0],
        'PaymentMethod_Mailed check': [1 if payment_method == 'Mailed check' else 0]
    })

    # Scale numerical features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    # Predict
    prediction = xgb_model.predict(input_data)[0]
    probability = xgb_model.predict_proba(input_data)[0][1]

    result = "Churn" if prediction == 1 else "No Churn"

    prediction_text = f"<strong>Prediction:</strong> {result}<br><strong>Probability of Churn:</strong> {probability:.2f}"

    return render_template('form.html', prediction_xgb=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
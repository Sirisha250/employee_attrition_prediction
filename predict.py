import pandas as pd
import joblib

# Load trained model, scaler, and feature selector
model = joblib.load("models/attrition_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_selector = joblib.load("models/feature_selector.pkl")  # Load feature selector

# Define expected features
expected_features = [
    "Age", "JobSatisfaction", "WorkLifeBalance", "MonthlyIncome", "YearsAtCompany",
    "PerformanceRating", "OverTime", "EducationLevel", "Gender", "Department", "NumProjects"
]

def preprocess_input(employee_data):
    """
    Preprocess input data:
    - Encode categorical features
    - Scale numerical features
    - Select the same features used during training
    """
    data = pd.DataFrame([employee_data], columns=expected_features)

    # Encode categorical features
    data["OverTime"] = data["OverTime"].map({"Yes": 1, "No": 0})
    data["EducationLevel"] = data["EducationLevel"].map({"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4})
    data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
    data["Department"] = data["Department"].map({"Sales": 1, "IT": 2, "HR": 3, "Finance": 4, "Marketing": 5})

    # Handle missing values
    data.fillna(0, inplace=True)

    # Scale numerical features
    data_scaled = scaler.transform(data)

    # Apply the same feature selection as in training
    data_selected = feature_selector.transform(data_scaled)

    return data_selected

def predict_attrition(employee_data):
    """
    Predict employee attrition and return result.
    """
    preprocessed_data = preprocess_input(employee_data)
    prediction = model.predict(preprocessed_data)[0]
    prediction_prob = model.predict_proba(preprocessed_data)[0]

    # Convert prediction: 0 -> "Yes" (Attrition), 1 -> "No" (No Attrition)
    result = {
        "Prediction": "Yes" if prediction == 0 else "No",
        "Confidence": f"{max(prediction_prob) * 100:.2f}%"
    }
    return result

# Sample employee data for testing
sample_employee = {
    "Age": 30,
    "JobSatisfaction": 2,
    "WorkLifeBalance": 3,
    "MonthlyIncome": 60000,
    "YearsAtCompany": 5,
    "PerformanceRating": 4,
    "OverTime": "No",
    "EducationLevel": "Master's",
    "Gender": "Male",
    "Department": "IT",
    "NumProjects": 4
}

if __name__ == "__main__":
    result = predict_attrition(sample_employee)
    print(f"Employee Attrition Prediction: {result['Prediction']} (Confidence: {result['Confidence']})")

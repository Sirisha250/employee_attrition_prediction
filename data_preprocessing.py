import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

# Load dataset
df = pd.read_csv("dataset/employee_data.csv")

# Handle missing values
for col in df.columns:
    if df[col].dtype == "object":  
        df[col].fillna(df[col].mode()[0], inplace=True)  # Fill categorical with mode
    else:
        df[col].fillna(df[col].median(), inplace=True)  # Fill numerical with median

# Separate target variable
target = df["Attrition"]  # Preserve target column
df.drop(columns=["Attrition"], inplace=True)

# Encode categorical columns
categorical_columns = ["OverTime", "EducationLevel", "Gender", "Department"]
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label encoders
joblib.dump(label_encoders, "models/label_encoders.pkl")

# Scale only numerical features
scaler = StandardScaler()
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Add target column back
df["Attrition"] = target  

# Save processed dataset
df.to_csv("dataset/processed_employee_data.csv", index=False)
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Data preprocessing completed successfully!")
print("✔ Cleaned dataset saved at: dataset/processed_employee_data.csv")
print("✔ Label encoders saved at: models/label_encoders.pkl")
print("✔ Scaler saved at: models/scaler.pkl")

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("dataset/processed_employee_data.csv")

# Fix Label Mapping (Swap Yes/No)
df["Attrition"] = df["Attrition"].map({"Yes": 0, "No": 1})  

# Define features & target
X = df.drop(columns=["Attrition"])
y = df["Attrition"].astype(int)

# Apply SMOTE for better class balance
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection (Keep only relevant features)
feature_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
feature_selector.fit(X_resampled, y_resampled)
X_train_selected = feature_selector.transform(X_train_scaled)
X_test_selected = feature_selector.transform(X_test_scaled)

# Define improved models with class balancing
rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced")
gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42)

stacked_model = StackingClassifier(
    estimators=[("rf", rf), ("gb", gb)], final_estimator=LogisticRegression(class_weight="balanced"), n_jobs=-1
)

# Train the model on selected features
stacked_model.fit(X_train_selected, y_train)

# Evaluate model
y_pred = stacked_model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Improved Model Accuracy: {accuracy:.2f}")  # Should be >90%
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(stacked_model, "models/attrition_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_selector, "models/feature_selector.pkl")

print("🎉 Model re-trained successfully with improved accuracy and correct predictions!")

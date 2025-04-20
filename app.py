from flask import Flask, render_template, request, redirect, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Database Configuration (SQLite)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///employee.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# User model for authentication
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Create the database if it doesn't exist
if not os.path.exists("employee.db"):
    with app.app_context():
        db.create_all()

# Load dataset dynamically
df = pd.read_csv("dataset/employee_data.csv")

# Convert categorical values to numerical
mappings = {
    "OverTime": {"Yes": 1, "No": 0},
    "Gender": {"Male": 1, "Female": 0},
    "Department": {"Sales": 0, "IT": 1, "HR": 2, "Finance": 3, "Marketing": 4},
    "EducationLevel": {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
}

for col, mapping in mappings.items():
    df[col] = df[col].map(mapping)

# Features and labels
X = df.drop(columns=["Attrition"])
y = df["Attrition"].map({"Yes": 1, "No": 0})

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model dynamically
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Signup Route
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect("/signup")

        if User.query.filter_by(email=email).first():
            flash("Email already registered!", "danger")
            return redirect("/signup")

        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
        new_user = User(email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Signup successful! Please login.", "success")
        return redirect("/login")

    return render_template("signup.html")

# Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session["user"] = email
            flash("Login successful!", "success")
            return redirect("/employee_attrition")
        flash("Invalid email or password", "danger")
        return redirect("/login")

    return render_template("login.html")

# Logout
@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully.", "info")
    return redirect("/")

# Employee Attrition Prediction Page
@app.route("/employee_attrition", methods=["GET"])
def employee_attrition():
    if "user" not in session:
        flash("Please log in first.", "danger")
        return redirect("/login")

    return render_template("employee_attrition.html")

@app.route("/predict_attrition", methods=["POST"])
def predict_attrition():
    if "user" not in session:
        flash("Please log in first.", "danger")
        return redirect("/login")

    try:
        print("📌 Received form data:", request.form)  # Debugging

        # Extract form data
        age = int(request.form["age"])
        job_satisfaction = int(request.form["job_satisfaction"])
        work_life_balance = int(request.form["work_life_balance"])
        monthly_income = int(request.form["monthly_income"])
        years_at_company = int(request.form["years_at_company"])
        performance_rating = int(request.form["performance_rating"])
        overtime = 1 if request.form["overtime"] == "Yes" else 0
        education_level = mappings["EducationLevel"].get(request.form["education_level"], 0)
        department = mappings["Department"].get(request.form["department"], 0)
        gender = 1 if request.form["gender"] == "Male" else 0
        num_projects = int(request.form["num_projects"])

        # Prepare input features as a DataFrame to maintain feature names
        input_features = pd.DataFrame([[age, job_satisfaction, work_life_balance, monthly_income,
                                        years_at_company, performance_rating, overtime, education_level,
                                        gender, department, num_projects]],
                                      columns=X.columns)  # Ensures correct column names

        # Scale the features
        scaled_features = scaler.transform(input_features)

        # Predict attrition
        prediction = model.predict(scaled_features)[0]
        prediction_text = "Yes" if prediction == 1 else "No"

        print("📌 Prediction:", prediction_text)  # Debugging

        return render_template("result_attrition.html",
                               prediction=prediction_text,
                               age=age,
                               job_satisfaction=job_satisfaction,
                               work_life_balance=work_life_balance,
                               monthly_income=monthly_income,
                               years_at_company=years_at_company,
                               performance_rating=performance_rating,
                               overtime="Yes" if overtime == 1 else "No",
                               education_level=request.form["education_level"],
                               gender="Male" if gender == 1 else "Female",
                               department=request.form["department"],
                               num_projects=num_projects)

    except Exception as e:
        print("❌ Error:", str(e))  # Debugging
        flash(f"Error processing data: {str(e)}", "danger")
        return redirect("/employee_attrition")

if __name__ == "__main__":
    app.run(debug=True)

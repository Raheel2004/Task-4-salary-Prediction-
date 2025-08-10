

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

#1. Choose a Dataset
dataset = pd.read_csv("https://raw.githubusercontent.com/Raheel2004/Task-4-salary-Prediction-/main/Employers_data.csv")  # Local file in repo

# Step 3: Data Preprocessing
# Check for nulls
print(dataset.isnull().sum())

# Encode categorical features
label_encoders = {}
for col in dataset.select_dtypes(include='object').columns:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])
    label_encoders[col] = le

# Scale numeric features
scaler = StandardScaler()
dataset[['Experience_Years', 'Salary']] = scaler.fit_transform(dataset[['Experience_Years', 'Salary']])

# Step 4: Feature Engineering (optional)
# Not required if dataset is clean and simple

# Step 5: Split Data
X = dataset.drop('Salary', axis=1)
y = dataset['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Models
lr = LinearRegression()
lr.fit(X_train, y_train)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Step 7: Evaluate Models
def evaluate(model):
    y_pred = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2 Score:", r2_score(y_test, y_pred))

print("Linear Regression Evaluation")
evaluate(lr)

print("Random Forest Regressor Evaluation")
evaluate(rf)

import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

st.title("💼 Salary Prediction App")

# Model load function
@st.cache_resource
def load_model():
    try:
        url = "https://github.com/Raheel2004/Task-4-salary-Prediction-/raw/main/salary_model.pkl"
        response = requests.get(url)
        response.raise_for_status()
        return joblib.load(BytesIO(response.content))
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None

model = load_model()

# User input fields
st.sidebar.header("Enter Employee Details")
age = st.sidebar.number_input("Age", 18, 65, 30)
experience = st.sidebar.number_input("Years of Experience", 0, 40, 5)
education = st.sidebar.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])

# Convert to DataFrame
input_df = pd.DataFrame({
    "age": [age],
    "experience": [experience],
    "education": [education]
})

# Prediction
if st.sidebar.button("Predict Salary"):
    if model:
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"💰 Predicted Salary: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.error("Model not available. Please check the model file.")

st.write("ℹ️ This app predicts employee salary using a trained ML model.")

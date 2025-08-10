

#Original file is located at
   # https://colab.research.google.com/drive/1wIstH96yX00hZ9gLbPfuVoHr9MaTeInN


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
import pandas as pd


# Update the path to your CSV file
# Make sure the file 'kaggle datasets.csv' is in the root of your Google Drive
# Load the datasets
import pandas as pd


dataset = pd.read_csv("https://raw.githubusercontent.com/Raheek2004/repo/main/data/Employers_data.csv")

dataset.head()

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

# Step 8: Streamlit App
# Save this section as app.py for deployment
"""
import streamlit as st
st.title("Salary Prediction App")

age = st.slider("Age", 20, 60)
experience = st.slider("Experience", 0, 40)
education = st.selectbox("Education Level", list(label_encoders['Education'].classes_))
skill = st.selectbox("Skill Level", list(label_encoders['Skill Level'].classes_))

# Convert input
input_df = pd.DataFrame({
    'Age': [age],
    'Experience': [experience],
    'Education': [label_encoders['Education'].transform([education])[0]],
    'Skill Level': [label_encoders['Skill Level'].transform([skill])[0]]
})

input_df[['Age', 'Experience']] = scaler.transform(input_df[['Age', 'Experience']])

prediction = rf.predict(input_df)[0]
st.success(f"Predicted Salary: ${prediction:,.2f}")
"""

# Step 9: Deployment
# Use 'render.yaml' or Render Dashboard to deploy the app



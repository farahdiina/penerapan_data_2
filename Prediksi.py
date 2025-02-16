import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("data_clean.csv")
    return df

df = load_data()

st.title("Student Dropout Prediction App")
st.sidebar.header("User Input Features")

input_data = {}
categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
numerical_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]

for col in categorical_columns:
    options = df[col].unique().tolist()
    input_data[col] = st.sidebar.selectbox(f"{col}", options)

for col in numerical_columns:
    min_val, max_val = float(df[col].min()), float(df[col].max())
    input_data[col] = st.sidebar.slider(f"{col}", min_value=min_val, max_value=max_val, value=min_val)

input_df = pd.DataFrame([input_data])

for col in categorical_columns:
    input_df[col] = encoder[col].transform(input_df[col])

input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

if st.sidebar.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    st.write(f"## Prediction: {'Dropout' if prediction == 1 else 'Continue'}")
    st.write(f"### Confidence: {max(prediction_proba) * 100:.2f}%")

st.subheader("Dataset Overview")
st.write(df.head())

fig = px.histogram(df, x='Status', title='Distribusi Status Mahasiswa')
st.plotly_chart(fig)

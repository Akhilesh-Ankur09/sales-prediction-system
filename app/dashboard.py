import streamlit as st
import pandas as pd
import joblib
import os


# Resolve project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rossmann_sales_model.pkl")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")


st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    layout="wide"
)

st.title("Retail Sales Forecasting System")
st.write("Predict weekly store sales using machine learning and real business drivers.")

# Load model
model = joblib.load(MODEL_PATH)

# Load feature template
X_template = pd.read_csv(os.path.join(DATA_PATH, "X_train.csv")).head(1)

# Business Inputs
st.sidebar.header("Business Inputs")

customers = st.sidebar.number_input("Customers", min_value=0, value=500)
day_of_week = st.sidebar.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7])
promo = st.sidebar.selectbox("Promotion Active", [0, 1])
open_store = st.sidebar.selectbox("Store Open", [0, 1])
school_holiday = st.sidebar.selectbox("School Holiday", [0, 1])
competition_distance = st.sidebar.number_input("Competition Distance (meters)", min_value=0, value=500)
promo2 = st.sidebar.selectbox("Promo2 Program Active", [0, 1])

# Build full feature vector from template
input_data = X_template.copy()

input_data["Customers"] = customers
input_data["DayOfWeek"] = day_of_week
input_data["Promo"] = promo
input_data["Open"] = open_store
input_data["SchoolHoliday"] = school_holiday
input_data["CompetitionDistance"] = competition_distance

# Optional fields (only if they exist in model)
if "Promo2" in input_data.columns:
    input_data["Promo2"] = promo2

st.subheader("Business Scenario")
st.dataframe(input_data[["Customers", "DayOfWeek", "Promo", "Open", "SchoolHoliday", "CompetitionDistance"]])

if st.button("Predict Weekly Sales"):
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Weekly Sales: ₹ {prediction:,.2f}")

    st.write(
        "This forecast is generated using a trained machine learning model on historical retail sales data. "
        "Adjust the business inputs to simulate different scenarios."
    )

import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    return df

@st.cache_resource
def train_model_cached(path):
    df = load_data(path)
    models = {}
    features = ["Day", "Month", "Year", "DayOfWeek", "WeekOfYear"]
    for product in df["Product line"].unique():
        pdf = df[df["Product line"] == product].copy()
        daily = pdf.groupby(["Date", "Day", "Month", "Year", "DayOfWeek", "WeekOfYear"])["Sales"].sum().reset_index()
        X = daily[features]
        y = daily["Sales"]
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X, y)
        models[product] = model
    return models

# Keep original name so app.py works without changes
def train_model(df):
    return train_model_cached("data/SuperMarket_Analysis.csv")

def predict_future(model, product, start_date, days):
    future_dates = pd.date_range(start=start_date, periods=days)
    future_df = pd.DataFrame({
        "Date": future_dates,
        "Day": future_dates.day,
        "Month": future_dates.month,
        "Year": future_dates.year,
        "DayOfWeek": future_dates.dayofweek,
        "WeekOfYear": future_dates.isocalendar().week.astype(int).values
    })
    features = ["Day", "Month", "Year", "DayOfWeek", "WeekOfYear"]
    future_df["Predicted Sales"] = model.predict(future_df[features])
    future_df["Predicted Sales"] = future_df["Predicted Sales"].clip(lower=0)
    return future_df
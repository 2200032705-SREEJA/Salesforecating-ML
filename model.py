import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    return df

def train_model(df):
    X = df[["Day", "Month", "Year"]]
    y = df["Sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def predict_and_plot(model, X_test, y_test):
    preds = model.predict(X_test)
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.title("Sales Prediction")
    import os
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/prediction_graph.png")
    return preds

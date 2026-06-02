import streamlit as st
import os
from model import load_data, train_model, predict_and_plot

st.set_page_config(page_title="Sales Forecasting ML", page_icon="📈")

st.title("📈 Sales Forecasting using XGBoost")

data = load_data("data/sales_data.csv")

st.subheader("Dataset Preview")
st.dataframe(data.head())

model, X_test, y_test = train_model(data)
predict_and_plot(model, X_test, y_test)

if os.path.exists("outputs/prediction_graph.png"):
    st.image(
        "outputs/prediction_graph.png",
        caption="Actual vs Predicted Sales"
    )
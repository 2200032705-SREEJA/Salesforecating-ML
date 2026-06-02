import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import load_data, train_model, predict_future
import datetime

st.set_page_config(page_title="Supermarket Sales Forecasting", page_icon="🛒", layout="wide")

st.title("🛒 Supermarket Sales Forecasting")
st.markdown("Predict future sales by product line using XGBoost ML model.")

# Load and train
data = load_data("data/SuperMarket_Analysis.csv")
models = train_model(data)

# ── Real-world brand labels ──────────────────────────────────────────────────
PRODUCT_LABELS = {
    "Electronic accessories": "📱 Electronics  (Apple · Samsung · boAt)",
    "Fashion accessories":    "👗 Fashion       (Zara · H&M · Levi's)",
    "Food and beverages":     "🍜 Food & Drinks (Maggi · Coca-Cola · Lay's)",
    "Health and beauty":      "💄 Health/Beauty (L'Oréal · Dove · Nivea)",
    "Home and lifestyle":     "🏠 Home          (IKEA · Philips · Prestige)",
    "Sports and travel":      "⚽ Sports        (Nike · Adidas · Decathlon)",
}
LABEL_TO_KEY = {v: k for k, v in PRODUCT_LABELS.items()}

product_lines   = sorted(data["Product line"].unique())
display_options = [PRODUCT_LABELS.get(p, p) for p in product_lines]

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Forecast Settings")

selected_label = st.sidebar.selectbox("Select Product Line", display_options)
product        = LABEL_TO_KEY.get(selected_label, selected_label)

future_days = st.sidebar.slider("Forecast how many days ahead?", 7, 90, 30)

# Default = today
start_date = st.sidebar.date_input("Start forecast from", value=datetime.date.today())

# ── Output Selector ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📤 Select Outputs to Display")

OUTPUT_OPTIONS = [
    "📊 Forecast Summary (KPIs)",
    "📈 Predicted Sales Over Time",
    "📊 Weekly Sales Forecast",
    "🏆 Product Line Comparison",
    "📋 Historical Sales Data",
    "📅 Forecast Table",
    "💡 Insights & Recommendations",
]

show_all = st.sidebar.checkbox("Show All", value=True)

if show_all:
    selected_outputs = OUTPUT_OPTIONS
else:
    selected_outputs = st.sidebar.multiselect(
        "Choose sections:",
        OUTPUT_OPTIONS,
        default=["📈 Predicted Sales Over Time"],
    )

def show(section):
    return section in selected_outputs

# ── Predict ──────────────────────────────────────────────────────────────────
forecast_df = predict_future(models[product], product, start_date, future_days)

# ── KPI metrics ──────────────────────────────────────────────────────────────
if show("📊 Forecast Summary (KPIs)"):
    st.subheader(f"📊 Forecast Summary for: **{selected_label}**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Forecast Days",       future_days)
    col2.metric("💰 Avg Predicted Sales", f"${forecast_df['Predicted Sales'].mean():.2f}")
    col3.metric("📈 Max Predicted Sales", f"${forecast_df['Predicted Sales'].max():.2f}")
    col4.metric("📉 Min Predicted Sales", f"${forecast_df['Predicted Sales'].min():.2f}")

# ── Line chart ───────────────────────────────────────────────────────────────
if show("📈 Predicted Sales Over Time"):
    st.subheader("📈 Predicted Sales Over Time")
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(forecast_df["Date"], forecast_df["Predicted Sales"],
             color="royalblue", linewidth=2, marker="o", markersize=3)
    ax1.fill_between(forecast_df["Date"], forecast_df["Predicted Sales"],
                     alpha=0.2, color="royalblue")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Predicted Sales ($)")
    ax1.set_title(f"Sales Forecast — {selected_label}")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

# ── Weekly bar chart ─────────────────────────────────────────────────────────
if show("📊 Weekly Sales Forecast"):
    st.subheader("📊 Weekly Sales Forecast (Bar Chart)")
    forecast_df["Week"] = forecast_df["Date"].dt.strftime("Week %U")
    weekly = forecast_df.groupby("Week")["Predicted Sales"].sum().reset_index()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(weekly["Week"], weekly["Predicted Sales"], color="coral", edgecolor="black")
    ax2.set_xlabel("Week")
    ax2.set_ylabel("Total Predicted Sales ($)")
    ax2.set_title("Weekly Sales Forecast")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# ── Product comparison ───────────────────────────────────────────────────────
# Always compute comp_df if needed for Insights or Comparison
need_comp = show("🏆 Product Line Comparison") or show("💡 Insights & Recommendations")
if need_comp:
    all_preds = {}
    for p in sorted(data["Product line"].unique()):
        df_p = predict_future(models[p], p, start_date, future_days)
        all_preds[p] = df_p["Predicted Sales"].mean()

    comp_df = pd.DataFrame(list(all_preds.items()), columns=["Product Line", "Avg Predicted Sales"])
    comp_df["Label"] = comp_df["Product Line"].map(PRODUCT_LABELS)
    comp_df = comp_df.sort_values("Avg Predicted Sales", ascending=True)

if show("🏆 Product Line Comparison"):
    st.subheader("🏆 Product Line Comparison (Avg Forecast)")
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    colors = ["green" if p == product else "steelblue" for p in comp_df["Product Line"]]
    bars = ax3.barh(comp_df["Label"], comp_df["Avg Predicted Sales"],
                    color=colors, edgecolor="black")
    ax3.set_xlabel("Avg Predicted Sales ($)")
    ax3.set_title("All Product Lines — Avg Predicted Sales")
    for bar, val in zip(bars, comp_df["Avg Predicted Sales"]):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"${val:.1f}", va="center")
    st.pyplot(fig3)

# ── Historical data ──────────────────────────────────────────────────────────
if show("📋 Historical Sales Data"):
    st.subheader("📋 Historical Sales Data")
    hist = (data[data["Product line"] == product]
            [["Date", "Sales", "Quantity", "Rating"]]
            .sort_values("Date"))
    st.dataframe(hist.tail(20), use_container_width=True)

# ── Forecast table ───────────────────────────────────────────────────────────
if show("📅 Forecast Table"):
    st.subheader("📅 Forecast Table")
    st.dataframe(
        forecast_df[["Date", "Predicted Sales"]]
        .assign(**{"Predicted Sales": forecast_df["Predicted Sales"].round(2)}),
        use_container_width=True,
    )

# ── Insights ─────────────────────────────────────────────────────────────────
if show("💡 Insights & Recommendations"):
    st.subheader("💡 Insights & Recommendations")
    best_key   = comp_df.iloc[-1]["Product Line"]
    worst_key  = comp_df.iloc[0]["Product Line"]
    best_label = PRODUCT_LABELS.get(best_key, best_key)
    worst_label= PRODUCT_LABELS.get(worst_key, worst_key)

    st.success(f"✅ **Best performing:** {best_label} — highest predicted sales")
    st.warning(f"⚠️ **Needs attention:** {worst_label} — lowest predicted sales")
    st.info(
        f"📌 **{selected_label}** — avg forecast: "
        f"${all_preds[product]:.2f}/day over the next {future_days} days"
    )

if not selected_outputs:
    st.info("👈 Please select at least one output from the sidebar to display.")
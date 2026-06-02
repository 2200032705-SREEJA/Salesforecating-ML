import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from model import load_data, train_model, predict_future
import datetime

st.set_page_config(page_title="Supermarket Sales Forecasting", page_icon="🛒", layout="wide")

# ── Load and train ────────────────────────────────────────────────────────────
data   = load_data("data/SuperMarket_Analysis.csv")
models = train_model(data)

# ── Brand labels ──────────────────────────────────────────────────────────────
PRODUCT_LABELS = {
    "Electronic accessories": "📱 Electronics  (Apple · Samsung · boAt)",
    "Fashion accessories":    "👗 Fashion       (Zara · H&M · Levi's)",
    "Food and beverages":     "🍜 Food & Drinks (Maggi · Coca-Cola · Lay's)",
    "Health and beauty":      "💄 Health/Beauty (L'Oréal · Dove · Nivea)",
    "Home and lifestyle":     "🏠 Home          (IKEA · Philips · Prestige)",
    "Sports and travel":      "⚽ Sports        (Nike · Adidas · Decathlon)",
}
LABEL_TO_KEY    = {v: k for k, v in PRODUCT_LABELS.items()}
product_lines   = sorted(data["Product line"].unique())
display_options = [PRODUCT_LABELS.get(p, p) for p in product_lines]

# ── Sidebar: settings ─────────────────────────────────────────────────────────
st.sidebar.header("🔍 Forecast Settings")
selected_label = st.sidebar.selectbox("Select Product Line", display_options)
product        = LABEL_TO_KEY.get(selected_label, selected_label)
future_days    = st.sidebar.slider("Forecast how many days ahead?", 7, 90, 30)
start_date     = st.sidebar.date_input("Start forecast from", value=datetime.date.today())

OUTPUT_OPTIONS = [
    "📊 Forecast Summary (KPIs)",
    "📈 Predicted Sales Over Time",
    "📊 Weekly Sales Forecast",
    "🏆 Product Line Comparison",
    "📋 Historical Sales Data",
    "📅 Forecast Table",
    "💡 Insights & Recommendations",
]

# ── Header: title LEFT, output selector RIGHT ─────────────────────────────────
hdr_left, hdr_right = st.columns([2, 1])
with hdr_left:
    st.title("🛒 Supermarket Sales Forecasting")
    st.markdown("Predict future sales by product line using XGBoost ML model.")
with hdr_right:
    st.markdown("### 📤 Select Outputs to Display")
    show_all = st.checkbox("Show All", value=True)
    if show_all:
        selected_outputs = OUTPUT_OPTIONS
    else:
        selected_outputs = st.multiselect(
            "Choose sections:", OUTPUT_OPTIONS,
            default=["📈 Predicted Sales Over Time"],
        )
    if not selected_outputs and not show_all:
        st.warning("No outputs selected.")

def show(section):
    return section in selected_outputs

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────────────────────
forecast_df = predict_future(models[product], product, start_date, future_days)

# ── Chart style helper ────────────────────────────────────────────────────────
def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#f8f9fc")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", color="#d0d3e0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

# ── KPI metrics ───────────────────────────────────────────────────────────────
if show("📊 Forecast Summary (KPIs)"):
    st.subheader(f"📊 Forecast Summary for: **{selected_label}**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Forecast Days",        future_days)
    col2.metric("💰 Avg Predicted Sales",  f"${forecast_df['Predicted Sales'].mean():.2f}")
    col3.metric("📈 Max Predicted Sales",  f"${forecast_df['Predicted Sales'].max():.2f}")
    col4.metric("📉 Min Predicted Sales",  f"${forecast_df['Predicted Sales'].min():.2f}")

# ── Line chart: Predicted Sales Over Time ─────────────────────────────────────
if show("📈 Predicted Sales Over Time"):
    st.subheader("📈 Predicted Sales Over Time")
    fig1, ax1 = plt.subplots(figsize=(13, 4.5))
    fig1.patch.set_facecolor("#ffffff")

    dates  = forecast_df["Date"]
    sales  = forecast_df["Predicted Sales"]
    ma7    = sales.rolling(7, min_periods=1).mean()

    ax1.fill_between(dates, sales, alpha=0.15, color="#2563eb")
    ax1.plot(dates, sales, color="#2563eb", linewidth=1.8,
             marker="o", markersize=4, label="Daily Forecast")
    ax1.plot(dates, ma7,   color="#f59e0b", linewidth=2,
             linestyle="--", label="7-day Moving Avg")

    # annotate max & min
    idx_max = sales.idxmax(); idx_min = sales.idxmin()
    ax1.annotate(f"Max ${sales[idx_max]:.0f}",
                 xy=(dates[idx_max], sales[idx_max]),
                 xytext=(0, 12), textcoords="offset points",
                 ha="center", fontsize=8, color="#16a34a",
                 arrowprops=dict(arrowstyle="-", color="#16a34a", lw=1))
    ax1.annotate(f"Min ${sales[idx_min]:.0f}",
                 xy=(dates[idx_min], sales[idx_min]),
                 xytext=(0, -18), textcoords="offset points",
                 ha="center", fontsize=8, color="#dc2626",
                 arrowprops=dict(arrowstyle="-", color="#dc2626", lw=1))

    style_ax(ax1,
             title=f"Sales Forecast — {selected_label}",
             xlabel="Date", ylabel="Predicted Sales ($)")
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax1.legend(fontsize=9)
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    st.pyplot(fig1)

# ── Weekly bar chart ──────────────────────────────────────────────────────────
if show("📊 Weekly Sales Forecast"):
    st.subheader("📊 Weekly Sales Forecast (Bar Chart)")
    forecast_df["Week"] = forecast_df["Date"].dt.strftime("Week %U")
    weekly = forecast_df.groupby("Week")["Predicted Sales"].sum().reset_index()

    fig2, ax2 = plt.subplots(figsize=(11, 4.5))
    fig2.patch.set_facecolor("#ffffff")

    n      = len(weekly)
    cmap   = plt.cm.get_cmap("YlOrRd", n)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    bars = ax2.bar(weekly["Week"], weekly["Predicted Sales"],
                   color=colors, edgecolor="white", linewidth=1.2, width=0.6)

    # value labels on top of each bar
    for bar, val in zip(bars, weekly["Predicted Sales"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                 f"${val:,.0f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#374151")

    style_ax(ax2, title="Weekly Sales Forecast",
             xlabel="Week", ylabel="Total Predicted Sales ($)")
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)

# ── Product comparison ────────────────────────────────────────────────────────
need_comp = show("🏆 Product Line Comparison") or show("💡 Insights & Recommendations")
if need_comp:
    all_preds = {}
    for p in sorted(data["Product line"].unique()):
        df_p = predict_future(models[p], p, start_date, future_days)
        all_preds[p] = df_p["Predicted Sales"].mean()

    comp_df = pd.DataFrame(list(all_preds.items()),
                           columns=["Product Line", "Avg Predicted Sales"])
    comp_df["Label"]  = comp_df["Product Line"].map(PRODUCT_LABELS)
    comp_df           = comp_df.sort_values("Avg Predicted Sales", ascending=True)

if show("🏆 Product Line Comparison"):
    st.subheader("🏆 Product Line Comparison (Avg Forecast)")
    fig3, ax3 = plt.subplots(figsize=(13, 5))
    fig3.patch.set_facecolor("#ffffff")

    bar_colors = ["#f59e0b" if p == product else "#3b82f6"
                  for p in comp_df["Product Line"]]
    bars = ax3.barh(comp_df["Label"], comp_df["Avg Predicted Sales"],
                    color=bar_colors, edgecolor="white", linewidth=1, height=0.6)

    for bar, val in zip(bars, comp_df["Avg Predicted Sales"]):
        ax3.text(bar.get_width() + 4, bar.get_y() + bar.get_height() / 2,
                 f"${val:,.1f}", va="center", fontsize=9, fontweight="bold",
                 color="#1f2937")

    # reference line = mean of all
    mean_val = comp_df["Avg Predicted Sales"].mean()
    ax3.axvline(mean_val, color="#6b7280", linestyle="--", linewidth=1.2,
                label=f"Avg all lines: ${mean_val:,.0f}")

    style_ax(ax3, title="All Product Lines — Avg Predicted Sales",
             xlabel="Avg Predicted Sales ($)", ylabel="")
    ax3.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax3.grid(axis="x", color="#d0d3e0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax3.grid(axis="y", visible=False)
    ax3.legend(fontsize=9)

    # legend patch for selected
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="#f59e0b", label="Selected product line"),
                  Patch(facecolor="#3b82f6", label="Other product lines")]
    ax3.legend(handles=legend_els, fontsize=9, loc="lower right")

    plt.tight_layout()
    st.pyplot(fig3)

# ── Historical data ───────────────────────────────────────────────────────────
if show("📋 Historical Sales Data"):
    st.subheader("📋 Historical Sales Data")
    hist = (data[data["Product line"] == product]
            [["Date", "Sales", "Quantity", "Rating"]]
            .sort_values("Date"))
    st.dataframe(hist.tail(20), use_container_width=True)

# ── Forecast table ────────────────────────────────────────────────────────────
if show("📅 Forecast Table"):
    st.subheader("📅 Forecast Table")
    st.dataframe(
        forecast_df[["Date", "Predicted Sales"]]
        .assign(**{"Predicted Sales": forecast_df["Predicted Sales"].round(2)}),
        use_container_width=True,
    )

# ── Insights ──────────────────────────────────────────────────────────────────
if show("💡 Insights & Recommendations"):
    st.subheader("💡 Insights & Recommendations")
    best_key    = comp_df.iloc[-1]["Product Line"]
    worst_key   = comp_df.iloc[0]["Product Line"]
    best_label  = PRODUCT_LABELS.get(best_key,  best_key)
    worst_label = PRODUCT_LABELS.get(worst_key, worst_key)

    st.success(f"✅ **Best performing:** {best_label} — highest predicted sales")
    st.warning(f"⚠️ **Needs attention:** {worst_label} — lowest predicted sales")
    st.info(
        f"📌 **{selected_label}** — avg forecast: "
        f"${all_preds[product]:.2f}/day over the next {future_days} days"
    )

if not selected_outputs:
    st.info("👈 Please select at least one output from the sidebar to display.")
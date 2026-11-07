
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ============================================================
# RETAIL SALES INTELLIGENCE - ARCTIC BLUE EDITION
# ============================================================

st.set_page_config(
    page_title="Retail Sales Intelligence",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sales.xlsx"
MODEL_PATH = BASE_DIR / "sales_model.pkl"

MODEL_COLUMNS = [
    "Month",
    "Year",
    "customer_age",
    "itemDescription",
    "final_price",
    "sales_event",
    "inventory_pressure"
]

REQUIRED_COLUMNS = MODEL_COLUMNS + ["units_sold"]

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March",
    4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September",
    10: "October", 11: "November", 12: "December"
}


# ============================================================
# 1. THEMES
# ============================================================

THEMES = {
    "Light": {
        "BG": "#F1F6FA",
        "CARD": "#FFFFFF",
        "INPUT": "#F5F8FB",
        "TEXT": "#20364A",
        "MUTED": "#64798A",
        "ACCENT": "#3679A8",
        "TEAL": "#36A6A0",
        "BORDER": "#DCE6EE",
        "GRID": "#E6EDF2",
        "SIDEBAR": "#FFFFFF"
    },
    "Dark": {
        "BG": "#111F2B",
        "CARD": "#1C3040",
        "INPUT": "#243B4D",
        "TEXT": "#F1F7FC",
        "MUTED": "#A9BDCC",
        "ACCENT": "#76B9E8",
        "TEAL": "#55C9B9",
        "BORDER": "#30495B",
        "GRID": "#374F60",
        "SIDEBAR": "#172A39"
    }
}


# ============================================================
# 2. LOAD DATA AND MODEL
# ============================================================

@st.cache_data
def load_data(path):
    return pd.read_excel(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not DATA_PATH.exists():
    st.error("sales.xlsx not found in the project folder.")
    st.info("Place sales.xlsx beside app.py.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("sales_model.pkl not found in the project folder.")
    st.info("Place sales_model.pkl beside app.py.")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
    model = load_model(str(MODEL_PATH))

except Exception as error:
    st.error("Unable to load project files.")
    st.exception(error)
    st.stop()


# ============================================================
# 3. DATA VALIDATION
# ============================================================

missing_columns = [
    col for col in REQUIRED_COLUMNS if col not in df.columns
]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()

if df.empty:
    st.error("Dataset is empty.")
    st.stop()

numeric_columns = [
    "Month", "Year", "customer_age",
    "final_price", "units_sold"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=REQUIRED_COLUMNS).copy()

if df.empty:
    st.error("No valid records found in the dataset.")
    st.stop()

df["Month"] = df["Month"].astype(int)
df["Year"] = df["Year"].astype(int)

if not df["Month"].between(1, 12).all():
    st.error("Month values must be between 1 and 12.")
    st.stop()

if df["itemDescription"].nunique() == 0:
    st.error("No products found in the dataset.")
    st.stop()

if not hasattr(model, "predict"):
    st.error("Saved model does not support prediction.")
    st.stop()


# ============================================================
# 4. SIDEBAR AND THEME SELECTION
# ============================================================

with st.sidebar:

    st.markdown("## ❄️ Arctic Analytics")
    st.caption("RETAIL SALES INTELLIGENCE")

    st.divider()

    theme_name = st.radio(
        "Appearance",
        ["Light", "Dark"],
        horizontal=True
    )

    C = THEMES[theme_name]

    st.divider()

    st.markdown("### Project Overview")
    st.write("Retail Store Sales Prediction")

    st.caption(
        "Machine learning powered retail demand "
        "forecasting and business analytics."
    )

    st.divider()

    st.markdown("### Dataset Information")

    # Separate metric cards prevent sidebar truncation.
    st.metric(
        label="Total Records",
        value=f"{len(df):,}"
    )

    st.metric(
        label="Total Products",
        value=f"{df['itemDescription'].nunique():,}"
    )

    st.caption(
        f"Data Period: {df['Year'].min()} - {df['Year'].max()}"
    )

    st.divider()

    st.markdown("### Technology Stack")
    st.write("• Python")
    st.write("• Pandas")
    st.write("• Scikit-learn")
    st.write("• Streamlit")
    st.write("• Matplotlib")


# ============================================================
# 5. PROFESSIONAL CSS
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {C["BG"]};
        color: {C["TEXT"]};
    }}

    [data-testid="stSidebar"] {{
        background-color: {C["SIDEBAR"]};
        border-right: 1px solid {C["BORDER"]};
    }}

    [data-testid="stSidebar"] * {{
        color: {C["TEXT"]};
    }}

    .block-container {{
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    h1 {{
        color: {C["TEXT"]} !important;
        font-size: 44px !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }}

    h2 {{
        color: {C["TEXT"]} !important;
        font-size: 29px !important;
        font-weight: 750 !important;
    }}

    h3 {{
        color: {C["TEXT"]} !important;
        font-size: 23px !important;
        font-weight: 700 !important;
    }}

    [data-testid="stCaptionContainer"] p {{
        color: {C["MUTED"]} !important;
        font-size: 15px !important;
    }}

    /* MAIN METRICS */

    [data-testid="stMetric"] {{
        background: {C["CARD"]};
        border: 1px solid {C["BORDER"]};
        border-radius: 16px;
        padding: 20px;
        min-height: 115px;
        box-shadow: 0 5px 18px rgba(26,55,75,0.05);
    }}

    [data-testid="stMetricLabel"] p {{
        color: {C["MUTED"]} !important;
        font-size: 16px !important;
        font-weight: 650 !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {C["TEXT"]} !important;
        font-size: 30px !important;
        font-weight: 800 !important;
        white-space: nowrap !important;
    }}

    /* SIDEBAR METRICS - FIXED WIDTH ISSUE */

    [data-testid="stSidebar"] [data-testid="stMetric"] {{
        padding: 14px !important;
        min-height: 85px !important;
        width: 100%;
        box-shadow: none;
    }}

    [data-testid="stSidebar"] [data-testid="stMetricValue"] {{
        font-size: 25px !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }}

    [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {{
        font-size: 15px !important;
    }}

    /* INPUT LABELS */

    [data-testid="stWidgetLabel"] p {{
        color: {C["TEXT"]} !important;
        font-size: 19px !important;
        font-weight: 700 !important;
    }}

    /* DROPDOWNS */

    div[data-baseweb="select"] > div {{
        min-height: 56px !important;
        background: {C["INPUT"]} !important;
        border: 1px solid {C["BORDER"]} !important;
        border-radius: 11px !important;
    }}

    div[data-baseweb="select"] * {{
        font-size: 18px !important;
        color: {C["TEXT"]} !important;
    }}

    div[data-baseweb="popover"] * {{
        font-size: 17px !important;
    }}

    /* NUMBER INPUT */

    .stNumberInput input {{
        min-height: 52px;
        background: {C["INPUT"]} !important;
        color: {C["TEXT"]} !important;
        border-radius: 10px !important;
        font-size: 18px !important;
    }}

    /* BUTTONS */

    .stButton button,
    .stFormSubmitButton button {{
        min-height: 56px;
        background: {C["ACCENT"]};
        color: white !important;
        border: none;
        border-radius: 11px;
        font-size: 18px;
        font-weight: 750;
        transition: 0.2s ease;
    }}

    .stButton button:hover,
    .stFormSubmitButton button:hover {{
        background: {C["TEAL"]};
        color: white !important;
        transform: translateY(-1px);
    }}

    /* CONTAINERS */

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {C["CARD"]};
        border: 1px solid {C["BORDER"]};
        border-radius: 16px;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {C["BORDER"]};
        border-radius: 12px;
    }}

    hr {{
        border-color: {C["BORDER"]};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 6. CHART FUNCTIONS
# ============================================================

def style_chart(fig, ax):

    fig.patch.set_facecolor(C["CARD"])
    ax.set_facecolor(C["CARD"])

    ax.tick_params(
        axis="both",
        colors=C["MUTED"],
        labelsize=10
    )

    ax.grid(
        color=C["GRID"],
        linestyle="--",
        linewidth=0.8,
        alpha=0.9
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(C["BORDER"])


def monthly_chart(data):

    monthly = (
        data.groupby("Month")["units_sold"]
        .mean()
        .reindex(range(1, 13))
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    style_chart(fig, ax)

    x = list(range(1, 13))
    values = monthly.values

    ax.plot(
        x, values,
        color=C["ACCENT"],
        linewidth=3,
        marker="o",
        markersize=7
    )

    ax.fill_between(
        x, values,
        color=C["ACCENT"],
        alpha=0.12
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MONTH_NAMES[i][:3] for i in x]
    )

    ax.set_title(
        "Monthly Demand Trend",
        loc="left",
        color=C["TEXT"],
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_ylabel("Average Units Sold", color=C["MUTED"])

    fig.tight_layout()
    return fig


def yearly_chart(data):

    yearly = (
        data.groupby("Year")["units_sold"]
        .mean()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    style_chart(fig, ax)

    bars = ax.bar(
        yearly.index.astype(str),
        yearly.values,
        color=C["TEAL"],
        width=0.55
    )

    ax.bar_label(
        bars,
        fmt="%.1f",
        padding=5,
        fontsize=10,
        color=C["TEXT"]
    )

    ax.set_title(
        "Year-wise Average Demand",
        loc="left",
        color=C["TEXT"],
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_ylabel("Average Units Sold", color=C["MUTED"])
    ax.margins(y=0.20)

    fig.tight_layout()
    return fig


def top_products_chart(data):

    top = (
        data.groupby("itemDescription")["units_sold"]
        .mean()
        .nlargest(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    style_chart(fig, ax)

    bars = ax.barh(
        top.index.astype(str),
        top.values,
        color=C["ACCENT"],
        height=0.65
    )

    ax.bar_label(
        bars,
        fmt="%.1f",
        padding=5,
        fontsize=10,
        color=C["TEXT"]
    )

    ax.set_title(
        "Top 10 Products by Average Units Sold",
        loc="left",
        color=C["TEXT"],
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Average Units Sold", color=C["MUTED"])
    ax.margins(x=0.17)

    fig.tight_layout()
    return fig


def forecast_units_chart(data):

    fig, ax = plt.subplots(figsize=(8, 4))
    style_chart(fig, ax)

    x = list(range(len(data)))
    values = data["Predicted Units"].values

    ax.plot(
        x, values,
        color=C["ACCENT"],
        linewidth=3,
        marker="o",
        markersize=7
    )

    ax.fill_between(
        x, values,
        color=C["ACCENT"],
        alpha=0.12
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        data["Period"],
        rotation=35,
        ha="right"
    )

    ax.set_title(
        "Monthly Predicted Demand",
        loc="left",
        color=C["TEXT"],
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    ax.set_ylabel("Predicted Units", color=C["MUTED"])

    fig.tight_layout()
    return fig


def forecast_revenue_chart(data):

    fig, ax = plt.subplots(figsize=(8, 4))
    style_chart(fig, ax)

    bars = ax.bar(
        data["Period"],
        data["Estimated Revenue"],
        color=C["TEAL"],
        width=0.6
    )

    ax.bar_label(
        bars,
        labels=[
            f"₹{value:,.0f}"
            for value in data["Estimated Revenue"]
        ],
        padding=4,
        fontsize=9,
        color=C["TEXT"]
    )

    ax.set_title(
        "Monthly Estimated Revenue",
        loc="left",
        color=C["TEXT"],
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    ax.set_ylabel("Revenue (₹)", color=C["MUTED"])

    ax.yaxis.set_major_formatter(
        mticker.StrMethodFormatter("₹{x:,.0f}")
    )

    ax.tick_params(axis="x", labelrotation=35)

    fig.tight_layout()
    return fig


# ============================================================
# 7. DASHBOARD HEADER
# ============================================================

st.markdown("# Retail Sales Intelligence")

st.caption(
    "Advanced retail demand forecasting and business analytics "
    "powered by machine learning."
)

st.divider()


# ============================================================
# 8. OVERVIEW METRICS
# ============================================================

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Records", f"{len(df):,}")

k2.metric(
    "Total Products",
    f"{df['itemDescription'].nunique():,}"
)

k3.metric(
    "Average Units Sold",
    f"{df['units_sold'].mean():.1f}"
)

k4.metric(
    "Average Product Price",
    f"₹{df['final_price'].mean():,.2f}"
)


# ============================================================
# 9. FORECAST CONFIGURATION
# ============================================================

st.divider()

st.markdown("## Sales Forecasting")

st.caption(
    "Generate a single-month prediction or forecast demand "
    "across a custom monthly period."
)

forecast_mode = st.radio(
    "Choose Forecast Type",
    ["Single Month", "Custom Date Range"],
    horizontal=True
)

products = sorted(
    df["itemDescription"].dropna().astype(str).unique().tolist()
)

age_min = int(df["customer_age"].min())
age_max = int(df["customer_age"].max())

price_min = float(df["final_price"].min())
price_max = float(df["final_price"].max())
price_default = float(df["final_price"].median())

if price_min >= price_max:
    st.error("Invalid product price range in dataset.")
    st.stop()

age_default = min(max(25, age_min), age_max)

min_year = int(df["Year"].min())
max_year = int(df["Year"].max())

year_options = list(range(min_year, max_year + 4))

with st.container(border=True):

    with st.form("forecast_form"):

        st.markdown("### Forecast Parameters")

        left, right = st.columns(2, gap="large")

        with left:

            selected_product = st.selectbox(
                "Select Product",
                products
            )

            customer_age = st.slider(
                "Customer Age",
                min_value=age_min,
                max_value=age_max,
                value=age_default
            )

            product_price = st.number_input(
                "Product Price (₹)",
                min_value=price_min,
                max_value=price_max,
                value=price_default,
                step=1.0,
                format="%.2f"
            )

        with right:

            festive_season = st.selectbox(
                "Festival Season",
                ["No", "Yes"]
            )

            inventory_pressure = st.selectbox(
                "Inventory Pressure",
                ["Low", "High"]
            )

            st.caption(
                "Selected conditions will be applied "
                "to every month in the forecast."
            )

        st.divider()

        if forecast_mode == "Single Month":

            st.markdown("### Select Forecast Month")

            m1, m2 = st.columns(2)

            with m1:
                start_month = st.selectbox(
                    "Month",
                    list(MONTH_NAMES.keys()),
                    format_func=lambda x: MONTH_NAMES[x]
                )

            with m2:
                start_year = st.selectbox(
                    "Year",
                    year_options,
                    index=year_options.index(max_year + 1)
                )

            end_month = start_month
            end_year = start_year

        else:

            st.markdown("### Select Custom Forecast Range")

            m1, m2 = st.columns(2)

            with m1:

                start_month = st.selectbox(
                    "Start Month",
                    list(MONTH_NAMES.keys()),
                    key="range_start_month",
                    format_func=lambda x: MONTH_NAMES[x]
                )

                start_year = st.selectbox(
                    "Start Year",
                    year_options,
                    index=year_options.index(max_year + 1),
                    key="range_start_year"
                )

            with m2:

                end_month = st.selectbox(
                    "End Month",
                    list(MONTH_NAMES.keys()),
                    index=5,
                    key="range_end_month",
                    format_func=lambda x: MONTH_NAMES[x]
                )

                end_year = st.selectbox(
                    "End Year",
                    year_options,
                    index=year_options.index(max_year + 1),
                    key="range_end_year"
                )

        st.divider()

        forecast_submitted = st.form_submit_button(
            "Generate Forecast",
            use_container_width=True
        )


# ============================================================
# 10. MONTH-WISE MACHINE LEARNING FORECAST
# ============================================================

if forecast_submitted:

    start_period = pd.Period(
        year=int(start_year),
        month=int(start_month),
        freq="M"
    )

    end_period = pd.Period(
        year=int(end_year),
        month=int(end_month),
        freq="M"
    )

    if end_period < start_period:

        st.error(
            "End month must be after or equal to start month."
        )

        st.session_state.pop("forecast_result", None)

    else:

        forecast_periods = pd.period_range(
            start=start_period,
            end=end_period,
            freq="M"
        )

        sales_event = (
            "Festival"
            if festive_season == "Yes"
            else "Normal"
        )

        forecast_rows = []

        try:

            for period in forecast_periods:

                model_input = pd.DataFrame([{
                    "Month": int(period.month),
                    "Year": int(period.year),
                    "customer_age": int(customer_age),
                    "itemDescription": selected_product,
                    "final_price": float(product_price),
                    "sales_event": sales_event,
                    "inventory_pressure": inventory_pressure
                }], columns=MODEL_COLUMNS)

                predicted_value = float(
                    model.predict(model_input)[0]
                )

                if pd.isna(predicted_value):
                    raise ValueError(
                        "Model returned an invalid prediction."
                    )

                predicted_units = max(0.0, predicted_value)

                estimated_revenue = (
                    predicted_units * float(product_price)
                )

                forecast_rows.append({
                    "Month": int(period.month),
                    "Year": int(period.year),
                    "Period": period.strftime("%b %Y"),
                    "Product": selected_product,
                    "Predicted Units": predicted_units,
                    "Estimated Revenue": estimated_revenue
                })

            forecast_df = pd.DataFrame(forecast_rows)

            st.session_state["forecast_result"] = {
                "data": forecast_df,
                "product": selected_product,
                "price": float(product_price),
                "event": sales_event,
                "inventory": inventory_pressure,
                "age": int(customer_age)
            }

        except Exception as error:

            st.session_state.pop("forecast_result", None)

            st.error("Unable to generate forecast.")
            st.exception(error)


# ============================================================
# 11. FORECAST RESULTS
# ============================================================

if "forecast_result" in st.session_state:

    result = st.session_state["forecast_result"]
    forecast_df = result["data"].copy()

    total_units = forecast_df["Predicted Units"].sum()
    total_revenue = forecast_df["Estimated Revenue"].sum()
    average_monthly_units = forecast_df["Predicted Units"].mean()

    st.divider()

    st.markdown("## Forecast Results")

    st.caption(
        f"Forecast for {result['product']} "
        f"across {len(forecast_df)} month(s)."
    )

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Total Predicted Units",
        f"{total_units:,.1f}"
    )

    r2.metric(
        "Total Estimated Revenue",
        f"₹{total_revenue:,.2f}"
    )

    r3.metric(
        "Average Monthly Demand",
        f"{average_monthly_units:,.1f}"
    )

    # NATIVE STREAMLIT FORECAST CARD

    with st.container(border=True):

        st.caption("FORECAST CONFIGURATION")
        st.subheader(result["product"])

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Product Price",
            f"₹{result['price']:,.2f}"
        )

        d2.metric(
            "Customer Age",
            result["age"]
        )

        d3.metric(
            "Sales Event",
            result["event"]
        )

        d4.metric(
            "Inventory",
            result["inventory"]
        )

        st.caption(
            f"Forecast Period: {forecast_df.iloc[0]['Period']} "
            f"to {forecast_df.iloc[-1]['Period']}"
        )

    # MONTH-WISE TABLE

    st.markdown("### Month-wise Forecast")

    display_df = forecast_df[
        ["Period", "Predicted Units", "Estimated Revenue"]
    ].copy()

    display_df["Predicted Units"] = (
        display_df["Predicted Units"].round(2)
    )

    display_df["Estimated Revenue"] = (
        display_df["Estimated Revenue"].map(
            lambda x: f"₹{x:,.2f}"
        )
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # FORECAST GRAPHS

    st.markdown("### Forecast Trend Analysis")

    chart_left, chart_right = st.columns(2, gap="large")

    with chart_left:

        fig = forecast_units_chart(forecast_df)

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    with chart_right:

        fig = forecast_revenue_chart(forecast_df)

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # CSV DOWNLOAD

    csv_data = forecast_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Complete Forecast Report",
        data=csv_data,
        file_name="retail_monthly_forecast.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# 12. HISTORICAL SALES ANALYTICS
# ============================================================

st.divider()

st.markdown("## Historical Sales Analytics")

st.caption(
    "Explore trends and patterns from your historical retail dataset."
)

left_chart, right_chart = st.columns(
    [1.35, 1],
    gap="large"
)

with left_chart:

    with st.container(border=True):

        fig = monthly_chart(df)

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

with right_chart:

    with st.container(border=True):

        fig = yearly_chart(df)

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


# ============================================================
# 13. TOP PERFORMING PRODUCTS
# ============================================================

st.markdown("### Top Performing Products")

with st.container(border=True):

    fig = top_products_chart(df)

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# 14. DATASET PREVIEW
# ============================================================

with st.expander("Explore Dataset Preview"):

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.caption(
    "RETAIL SALES INTELLIGENCE | ARCTIC BLUE EDITION"
)

st.caption(
    "Built using Python, Machine Learning and Streamlit."
)

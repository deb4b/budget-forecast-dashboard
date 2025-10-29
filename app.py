import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

st.set_page_config(page_title="Budget Forecast Dashboard", layout="wide")

st.title("📊 Budget Forecast & Accrual Insights Dashboard")
st.markdown("Upload your SAP data (.csv) to explore spending, forecast trends, and detect accrual requirements.")

# ------------------- Upload CSV -------------------
uploaded_file = st.file_uploader("📁 Upload your SAP Data", type=["csv"])

if uploaded_file is not None:

    # ------------------- Load & Prepare -------------------
    df = pd.read_csv(uploaded_file)
    df['MonthDate'] = pd.to_datetime(df['MonthDate'], errors='coerce')
    df = df.sort_values('MonthDate')

    # Add AccrualStatus
    df['AccrualStatus'] = df['Total SAP Data'].apply(
        lambda x: "❌ Accrue Required" if x < 0 else "✅ On Track"
    )

    st.success("✅ Data uploaded successfully!")

    # ------------------- Sidebar Filters -------------------
    st.sidebar.header("🔍 Filter Options")

    selected_year = st.sidebar.multiselect(
        "Select Year:",
        options=sorted(df['MonthDate'].dt.year.unique()),
        default=sorted(df['MonthDate'].dt.year.unique())
    )

    selected_pmn = st.sidebar.multiselect(
        "Select PMN:",
        options=df['PMN'].unique(),
        default=df['PMN'].unique()
    )

    # Filter dataframe
    df_filtered = df[
        (df['MonthDate'].dt.year.isin(selected_year)) &
        (df['PMN'].isin(selected_pmn))
    ]

    # ------------------- Aggregate Monthly -------------------
    monthly = df_filtered.groupby('MonthDate')['Total SAP Data'].sum()

    # ------------------- KPI Metrics -------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("📌 Data Points", len(df_filtered))
    col2.metric("📊 Avg SAP", f"{monthly.mean():,.2f}")
    col3.metric("❌ Accrual Cases", (df_filtered['AccrualStatus'] == "❌ Accrue Required").sum())

    # ------------------- Historical Trend -------------------
    st.subheader("📈 Historical SAP Trend")
    fig_hist = go.Figure()
    colors = ['red' if v < 0 else 'green' for v in monthly.values]

    fig_hist.add_trace(go.Scatter(
        x=monthly.index.strftime("%b-%Y"),
        y=monthly.values,
        mode='lines+markers',
        name='Actual',
        marker=dict(color=colors, size=8)
    ))
    fig_hist.update_layout(xaxis_title="Month-Year", yaxis_title="Total SAP Data", hovermode="x unified")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()
    st.subheader("📌 Forecasting (ETS Model)")

    # ------------------- Forecasting -------------------
    if len(monthly) > 6:
        model = ExponentialSmoothing(monthly, trend='add', seasonal=None).fit()
        forecast_periods = 6
        forecast = model.forecast(forecast_periods)

        # Fix RangeIndex: assign proper datetime index
        last_date = monthly.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1),
            periods=forecast_periods,
            freq='MS'
        )
        forecast.index = forecast_dates

        # Align for metrics
        aligned_actual, aligned_pred = monthly.align(model.fittedvalues, join='inner')
        mae = mean_absolute_error(aligned_actual, aligned_pred)
        rmse = np.sqrt(mean_squared_error(aligned_actual, aligned_pred))  # fixed for old sklearn

        colA, colB = st.columns(2)
        colA.metric("MAE", f"{mae:,.2f}")
        colB.metric("RMSE", f"{rmse:,.2f}")

        # ------------------- Forecast Plot -------------------
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=monthly.index.strftime("%b-%Y"),
            y=monthly.values,
            mode='lines+markers',
            name='Actual'
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast.index.strftime("%b-%Y"),
            y=forecast.values,
            mode='lines+markers',
            name=f'{forecast_periods}-Month Forecast'
        ))
        fig_forecast.update_layout(
            title="Actual vs Forecast (ETS)",
            xaxis_title="Month-Year",
            yaxis_title="Total SAP Data",
            hovermode="x unified"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        # ------------------- Accrual Detection -------------------
        st.divider()
        st.subheader("🚨 Accrual Detection Table")
        accrual_table = df_filtered[df_filtered['AccrualStatus'] == "❌ Accrue Required"]
        if accrual_table.empty:
            st.success("✅ No accruals required.")
        else:
            st.error(f"⚠️ {len(accrual_table)} accrual cases detected!")
            st.dataframe(accrual_table[['MonthDate','PMN','Total SAP Data','AccrualStatus']], use_container_width=True)

        # ------------------- Scenario Planner -------------------
        st.divider()
        st.subheader("🧮 Scenario Planner")
        adjust = st.slider("Adjust budget by (%)", -50, 50, 0)
        scenario = forecast * (1 + adjust / 100)
        st.write("📍 Updated Forecast with Adjustment:")
        st.write(scenario)

    else:
        st.warning("🚫 Not enough historical data for forecasting (need >6 months).")

else:
    st.info("⬆️ Upload CSV to start analysis.")

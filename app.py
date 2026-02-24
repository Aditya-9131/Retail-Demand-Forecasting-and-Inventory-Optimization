import streamlit as st
import pandas as pd
import plotly.express as px
import os

from src.main import main as run_pipeline

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

st.title("Retail Demand Forecasting and Inventory Optimization Dashboard")
st.markdown("---")

run_button = st.sidebar.button("Run Full ML Pipeline")

if run_button:
    with st.spinner("Generating data, cleaning, and model training..."):
        run_pipeline()
        st.success("Pipeline executed successfully!")

st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Model Forecasts", "Inventory Insights"])

# Paths
eda_dir = 'output/eda'
models_dir = 'output/models'
inventory_dir = 'output/inventory'
cleaned_data_path = 'data/cleaned_retail_data.csv'
adv_pred_path = os.path.join(models_dir, 'advanced_predictions.csv')
inv_adjust_path = os.path.join(inventory_dir, 'store_product_stock_adjustments.csv')

def load_data(filepath):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

if page == "EDA":
    st.header("Exploratory Data Analysis")
    df = load_data(cleaned_data_path)
    if df is not None:
        st.subheader("Cleaned Dataset Sample")
        st.dataframe(df.head())
        
        # Display charts
        st.subheader("Overall Daily Demand Network-wide")
        df['Date'] = pd.to_datetime(df['Date'])
        daily = df.groupby('Date')['Demand'].sum().reset_index()
        fig1 = px.line(daily, x='Date', y='Demand', title='Total Daily Demand')
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Demand Spread by Store")
        fig2 = px.box(df, x='Store_ID', y='Demand', title='Demand by Store')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Data not found. Please run the pipeline first via the sidebar.")

elif page == "Model Forecasts":
    st.header("Forecasting Models")
    df_pred = load_data(adv_pred_path)
    if df_pred is not None:
        df_pred['Date'] = pd.to_datetime(df_pred['Date'])
        daily_pred = df_pred.groupby('Date')[['Demand', 'Forecast']].sum().reset_index()
        
        st.subheader("Actual Demand vs Forecast (Advanced Model Test Set)")
        fig = px.line(daily_pred, x='Date', y=['Demand', 'Forecast'], 
                      title='Network Wide Demand vs Model Forecast', 
                      labels={'value': 'Units', 'variable': 'Metric'})
        st.plotly_chart(fig, use_container_width=True)
        
        if os.path.exists(os.path.join(models_dir, 'feature_importances.png')):
            st.subheader("Feature Importances")
            st.image(os.path.join(models_dir, 'feature_importances.png'), caption='Random Forest Importances')
    else:
        st.warning("Model predictions not found. Please run the pipeline first.")

elif page == "Inventory Insights":
    st.header("Inventory Optimization Insights")
    df_inv = load_data(inv_adjust_path)
    if df_inv is not None:
        st.subheader("Inventory Replenishment & Surplus Adjustments")
        st.dataframe(df_inv.sort_values(by='Discrepancy', ascending=False))
        
        st.markdown('''
        * **Positive Discrepancy**: Model forecasts higher demand than currently actualized in the test set. Increase safety stock for these Store/Product pairs.
        * **Negative Discrepancy**: Historical demand is higher than future predicted sales. Reduce inventory depth.
        ''')
        
        report_text = ""
        report_path = os.path.join(inventory_dir, 'inventory_report.txt')
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_text = f.read()
            st.text_area("Inventory Consultant Summary Report", report_text, height=200)

    else:
         st.warning("Inventory records not found. Please run the pipeline first.")


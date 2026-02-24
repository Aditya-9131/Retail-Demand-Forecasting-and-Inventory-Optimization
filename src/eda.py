import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(input_path, output_dir):
    print(f"Loading data for EDA from {input_path}...")
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Total Daily Demand Trend
    plt.figure(figsize=(12, 6))
    daily_sales = df.groupby('Date')['Demand'].sum().reset_index()
    sns.lineplot(data=daily_sales, x='Date', y='Demand')
    plt.title('Total Daily Demand Over Time')
    plt.savefig(os.path.join(output_dir, 'total_daily_demand.png'))
    plt.close()
    
    # 2. Demand Distribution by Store
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='Store_ID', y='Demand')
    plt.title('Demand Distribution by Store')
    plt.savefig(os.path.join(output_dir, 'store_demand_dist.png'))
    plt.close()
    
    # 3. Monthly Seasonality
    plt.figure(figsize=(10, 5))
    monthly_sales = df.groupby('Month')['Demand'].mean().reset_index()
    sns.barplot(data=monthly_sales, x='Month', y='Demand')
    plt.title('Average Demand by Month')
    plt.savefig(os.path.join(output_dir, 'monthly_seasonality.png'))
    plt.close()

    print(f"EDA plots saved to {output_dir}")

if __name__ == '__main__':
    perform_eda('data/cleaned_retail_data.csv', 'output/eda')

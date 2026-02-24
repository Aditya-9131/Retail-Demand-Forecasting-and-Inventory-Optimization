import os
import pandas as pd
import numpy as np

def generate_synthetic_data(file_path):
    print("Generating synthetic retail data...")
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    stores = ['Store_A', 'Store_B', 'Store_C']
    products = ['Product_1', 'Product_2', 'Product_3']
    
    data = []
    for store in stores:
        for product in products:
            base_demand = np.random.randint(50, 200)
            trend = np.linspace(0, 50, len(dates))
            seasonality = 30 * np.sin(2 * np.pi * dates.dayofyear / 365)
            noise = np.random.normal(0, 10, len(dates))
            
            promotions = np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1])
            holidays = np.random.choice([0, 1], size=len(dates), p=[0.98, 0.02])
            
            demand = base_demand + trend + seasonality + 50 * promotions + 100 * holidays + noise
            demand = np.maximum(0, demand).astype(int)
            
            df_temp = pd.DataFrame({
                'Date': dates,
                'Store_ID': store,
                'Product_ID': product,
                'Demand': demand,
                'Promotion': promotions,
                'Holiday': holidays
            })
            data.append(df_temp)
            
    df = pd.concat(data, ignore_index=True)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Data successfully saved to {file_path}")
    return df

if __name__ == '__main__':
    generate_synthetic_data('data/raw_retail_data.csv')

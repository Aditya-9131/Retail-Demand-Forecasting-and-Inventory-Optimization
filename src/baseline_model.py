import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def run_baseline(input_path, output_dir):
    print(f"Loading data from {input_path} for Baseline Modeling...")
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Aggregate to daily total network sales for baseline
    daily_sales = df.groupby('Date')['Demand'].sum().reset_index()
    daily_sales.set_index('Date', inplace=True)
    
    train_size = int(len(daily_sales) * 0.8)
    train, test = daily_sales.iloc[:train_size], daily_sales.iloc[train_size:]
    
    # Using Exponential Smoothing as a Baseline with weekly seasonality for faster fitting
    model = ExponentialSmoothing(train['Demand'], trend='add', seasonal='add', seasonal_periods=7)
    fitted_model = model.fit()
    
    predictions = fitted_model.forecast(len(test))
    
    mae = mean_absolute_error(test['Demand'], predictions)
    rmse = np.sqrt(mean_squared_error(test['Demand'], predictions))
    
    print(f"Baseline Model MAE: {mae:.2f}")
    print(f"Baseline Model RMSE: {rmse:.2f}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Demand'], label='Train')
    plt.plot(test.index, test['Demand'], label='Test')
    plt.plot(test.index, predictions, label='Forecast')
    plt.legend()
    plt.title('Baseline Model Forecast vs Actuals')
    
    plot_path = os.path.join(output_dir, 'baseline_forecast.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Save predictions
    test_results = test.copy()
    test_results['Forecast'] = predictions
    test_results.to_csv(os.path.join(output_dir, 'baseline_predictions.csv'))
    
    return predictions, rmse

if __name__ == '__main__':
    run_baseline('data/cleaned_retail_data.csv', 'output/models')

import pandas as pd
import os

def clean_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Handle missing values
    df.ffill(inplace=True)
    
    # Convert dates properly
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Add date features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    
    # Ensure no negative demand
    df['Demand'] = df['Demand'].apply(lambda x: max(0, x))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    return df

if __name__ == '__main__':
    clean_data('data/raw_retail_data.csv', 'data/cleaned_retail_data.csv')

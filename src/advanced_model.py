import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def run_advanced_model(input_path, output_dir):
    print(f"Loading data from {input_path} for Advanced Modeling...")
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Feature Engineering
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # Encoding Categorical Variables
    df_encoded = pd.get_dummies(df, columns=['Store_ID', 'Product_ID'])
    
    # Sort chronological
    df_encoded.sort_values('Date', inplace=True)
    
    # Train-test split (80/20)
    train_size = int(len(df_encoded) * 0.8)
    
    features = [col for col in df_encoded.columns if col not in ['Date', 'Demand']]
    X = df_encoded[features]
    y = df_encoded['Demand']
    
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Random Forest Model
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    predictions = rf.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"Advanced Model (Random Forest) MAE: {mae:.2f}")
    print(f"Advanced Model (Random Forest) RMSE: {rmse:.2f}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature Importances Plot
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.head(10).values, y=importances.head(10).index)
    plt.title('Top 10 Feature Importances in Demand Forecasting')
    plt.savefig(os.path.join(output_dir, 'feature_importances.png'))
    plt.close()
    
    # Save test predictions with actuals for later inventory optimization
    df_test = df.iloc[train_size:].copy()
    df_test['Forecast'] = predictions
    df_test.to_csv(os.path.join(output_dir, 'advanced_predictions.csv'), index=False)
    
    return predictions, rmse

if __name__ == '__main__':
    run_advanced_model('data/cleaned_retail_data.csv', 'output/models')

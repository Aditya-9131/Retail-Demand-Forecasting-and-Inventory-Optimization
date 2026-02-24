import os
from src.data_generation import generate_synthetic_data
from src.data_cleaning import clean_data
from src.eda import perform_eda
from src.baseline_model import run_baseline
from src.advanced_model import run_advanced_model
from src.inventory_optimization import generate_inventory_insights

def main():
    print("Welcome to the Retail Demand Forecasting Pipeline!")
    
    # Paths configuration
    raw_data_path = 'data/raw_retail_data.csv'
    cleaned_data_path = 'data/cleaned_retail_data.csv'
    eda_out_dir = 'output/eda'
    models_out_dir = 'output/models'
    inventory_out_dir = 'output/inventory'
    
    # 1. Week 1: Data Generation
    generate_synthetic_data(raw_data_path)
    
    # 2. Week 2: Data Cleaning
    clean_data(raw_data_path, cleaned_data_path)
    
    # 3. Week 3: EDA
    perform_eda(cleaned_data_path, eda_out_dir)
    
    # 4. Week 4: Baseline Model
    run_baseline(cleaned_data_path, models_out_dir)
    
    # 5. Week 5: Advanced Model
    adv_predictions_file = os.path.join(models_out_dir, 'advanced_predictions.csv')
    run_advanced_model(cleaned_data_path, models_out_dir)
    
    # 6. Week 6: Inventory Insights
    generate_inventory_insights(adv_predictions_file, inventory_out_dir)
    
    print("\nPipeline completed successfully! Check the 'output' directory.")

if __name__ == '__main__':
    # Make sure execution is in the right dir
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('output'):
        os.makedirs('output')
    main()

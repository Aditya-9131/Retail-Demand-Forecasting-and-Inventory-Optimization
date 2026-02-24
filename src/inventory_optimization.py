import pandas as pd
import numpy as np
import os

def generate_inventory_insights(input_path, output_dir):
    print(f"Loading predictions from {input_path} for Inventory Optimization...")
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Simple logic: Safety Stock = Z * StdDev_Demand + LeadTimeDemand
    # Assuming Lead Time = 7 days
    std_dev_demand = df['Demand'].std()
    
    # 95% service level -> Z = 1.65
    safety_stock = 1.65 * std_dev_demand
    
    # Aggregate weekly demand
    df['Week'] = df['Date'].dt.isocalendar().week
    weekly_demand = df.groupby('Week')['Demand'].sum().mean()
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'inventory_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("Inventory Optimization Insights Report\n")
        f.write("======================================\n\n")
        f.write(f"Estimated Network-wide Safety Stock Required: {safety_stock:.2f} units\n")
        f.write(f"Average Weekly Operations Demand: {weekly_demand:.2f} units\n\n")
        f.write("Optimization Recommendations:\n")
        f.write("1. Use Advanced Model forecasts to dynamically adjust safety stock.\n")
        f.write("2. Identify specific slow-moving products and reduce baseline reorder levels.\n")
        f.write("3. Allocate stock closer to high-demand stores as predicted by store-level features.\n")
        
    print(f"Inventory Insights compiled and saved to {report_path}")
    
    # Generate store-product level insights
    detailed_insight = df.groupby(['Store_ID', 'Product_ID'])[['Demand', 'Forecast']].sum().reset_index()
    detailed_insight['Discrepancy'] = detailed_insight['Forecast'] - detailed_insight['Demand']
    detailed_insight.to_csv(os.path.join(output_dir, 'store_product_stock_adjustments.csv'), index=False)
    
if __name__ == '__main__':
    generate_inventory_insights('output/models/advanced_predictions.csv', 'output/inventory')

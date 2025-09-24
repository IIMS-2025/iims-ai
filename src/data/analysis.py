import pandas as pd

def analyze_synthetic_data(df):
    """Analyze the generated synthetic data for relationships."""
    print("=== SYNTHETIC DATA ANALYSIS ===\n")
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Tenants: {df['tenant_id'].nunique()}")
    print(f"Locations: {df['location_id'].nunique()}")
    print(f"Products: {df['product_id'].nunique()}")
    print(f"Average Daily Sales per Product: {df['target_sales_qty'].mean():.1f}")
    
    print("\n--- RELATIONSHIPS VERIFICATION ---")
    
    # Price-Sales correlation
    price_sales_corr = df.groupby('product_id').agg({
        'current_price': 'mean', 
        'target_sales_qty': 'mean'
    }).corr().iloc[0,1]
    print(f"Price-Sales Correlation: {price_sales_corr:.3f}")
    
    # Holiday effect
    holiday_avg = df[df['is_holiday']]['target_sales_qty'].mean()
    regular_avg = df[~df['is_holiday']]['target_sales_qty'].mean()
    print(f"Holiday vs Regular Sales: {holiday_avg:.1f} vs {regular_avg:.1f} ({holiday_avg/regular_avg:.2f}x)")
    
    # Day of week patterns
    dow_sales = df.groupby('day_of_week')['target_sales_qty'].mean().sort_values(ascending=False)
    print(f"Day of Week Sales (highest to lowest):")
    for day, sales in dow_sales.items():
        print(f"  {day}: {sales:.1f}")
    
    return df


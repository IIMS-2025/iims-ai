from generation import generate_realistic_sales_data
from analysis import analyze_synthetic_data
import os

if __name__ == '__main__':
    synthetic_df = generate_realistic_sales_data(num_days=730, num_locations=9, num_products=50, num_tenants=3)
    analyzed_df = analyze_synthetic_data(synthetic_df)
    
    print("\n=== SAMPLE DATA ===")
    print(analyzed_df.head(10))
    
    print(f"\n=== TENANT-PRODUCT VERIFICATION ===")
    tenant_product_check = analyzed_df.groupby('tenant_id')['product_id'].apply(lambda x: sorted(x.unique()))
    for tenant_id, products in tenant_product_check.items():
        print(f"Tenant {tenant_id}: Products {products[0]}-{products[-1]} (Total: {len(products)})")
    
    # Create data/datasets folder if it doesn't exist
    os.makedirs('data/datasets', exist_ok=True)
    
    # Save to data/datasets folder
    file_path = 'data/datasets/enhanced_synthetic_sales_data.csv'
    synthetic_df.to_csv(file_path, index=False)
    print(f"\nEnhanced dataset saved to '{file_path}'")

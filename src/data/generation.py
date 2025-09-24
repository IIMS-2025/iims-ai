import pandas as pd
import numpy as np

def generate_realistic_sales_data(num_days=365, num_locations=9, num_products=50, num_tenants=5):
    """
    Generates a highly realistic synthetic sales dataset with enhanced relationships.

    Args:
        num_days (int): The number of days to simulate.
        num_locations (int): The number of unique restaurant locations.
        num_products (int): The number of unique products PER TENANT.
        num_tenants (int): The number of unique tenants/restaurant chains.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the synthetic data.
    """

    # --- 1. SETUP CORE DATA ---
    dates = pd.date_range(start='2024-01-01', periods=num_days)
    tenant_ids = np.arange(1001, 1001 + num_tenants)
    location_ids = np.arange(101, 101 + num_locations)
    
    # Create tenant-specific product ranges (non-overlapping)
    tenant_product_mapping = {}
    product_info_all = pd.DataFrame()
    
    for i, tenant_id in enumerate(tenant_ids):
        # Each tenant gets their own range of product IDs
        start_product_id = 201 + (i * num_products)
        end_product_id = start_product_id + num_products
        tenant_product_ids = np.arange(start_product_id, end_product_id)
        tenant_product_mapping[tenant_id] = tenant_product_ids
        
        # Create product info for this tenant
        tenant_products = pd.DataFrame({
            'product_id': tenant_product_ids,
            'tenant_id': tenant_id,
            'product_category': np.random.choice(
                ['Beverage', 'Appetizer', 'Main Course', 'Dessert'], 
                size=num_products, 
                p=[0.3, 0.2, 0.4, 0.1]
            ),
            'current_price': np.random.uniform(50.0, 450.0, size=num_products).round(2),
            'last_price_change_date': pd.NaT,
            'seasonality_factor': np.random.uniform(0.85, 1.15, size=num_products),
            'weather_sensitivity': np.random.uniform(0.1, 0.4, size=num_products),
            'location_popularity': np.random.uniform(0.9, 1.1, size=num_products)
        }).set_index('product_id')
        
        # Add to the overall product info
        product_info_all = pd.concat([product_info_all, tenant_products])
    
    # Location characteristics
    location_info = pd.DataFrame({
        'location_id': location_ids,
        'location_type': np.random.choice(['Urban', 'Suburban', 'Mall', 'Tourist'], size=num_locations),
        'base_traffic': np.random.uniform(0.8, 1.3, size=num_locations), # Reduced from 0.7-1.5
        'weekend_boost': np.random.uniform(1.1, 1.4, size=num_locations) # Reduced from 1.1-1.6
    }).set_index('location_id')

    # --- 2. SIMULATE DAY-BY-DAY ---
    records = []
    daily_sales = {} # To store sales history
    inventory_stockout = {} # Track stockouts
    marketing_end_date = None # Track when current campaign ends

    for i, date in enumerate(dates):
        day_of_week = date.day_name()
        is_weekend = day_of_week in ['Saturday', 'Sunday']
        is_holiday = is_weekend or (np.random.rand() < 0.02)
        month = date.month
        
        # --- ENHANCED EXTERNAL FACTORS ---
        
        # Weather effect (seasonal pattern + random variation)
        weather_base = 0.5 * np.sin(2 * np.pi * (date.dayofyear - 81) / 365) + 0.5 # Peak in summer
        daily_weather = weather_base + np.random.normal(0, 0.2) # Add noise
        daily_weather = np.clip(daily_weather, 0, 1) # Keep in [0,1] range
        
        # Marketing campaigns (5% chance of starting, lasts 3-7 days)
        is_marketing_active = False
        if marketing_end_date is None or date > marketing_end_date:
            # No active campaign, chance to start new one
            if np.random.rand() < 0.05:  # 5% chance to start campaign
                campaign_duration = np.random.randint(3, 8)
                marketing_end_date = date + pd.Timedelta(days=campaign_duration-1)
                is_marketing_active = True
        else:
            # Campaign is currently active
            is_marketing_active = True
        
        # Economic cycles (monthly variation)
        economic_factor = 1.0 + 0.1 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 0.05)
        economic_factor = np.clip(economic_factor, 0.8, 1.2)

        # --- SELECT TENANT, PRODUCTS & LOCATIONS ---
        # First select a tenant for this day's transactions
        tenant_id = np.random.choice(tenant_ids)
        tenant_products = tenant_product_mapping[tenant_id]
        
        # Select products from this tenant's catalog only
        num_products_sold = np.random.randint(20, min(41, len(tenant_products) + 1))
        products_sold_today = np.random.choice(tenant_products, size=num_products_sold, replace=False)
        
        for product_id in products_sold_today:
            # Location selection with weighted probability
            location_weights = location_info['base_traffic'].values
            location_id = np.random.choice(location_ids, p=location_weights/location_weights.sum())
            
            # --- ENHANCED PRICING LOGIC ---
            competitor_price_change = np.random.rand() < 0.005 # 0.5% chance competitor changes price
            demand_based_pricing = np.random.rand() < 0.003 # 0.3% chance of demand-based adjustment
            
            if np.random.rand() < 0.01 or competitor_price_change or demand_based_pricing:
                if competitor_price_change:
                    price_change_factor = np.random.uniform(0.97, 1.03) # Competitive response
                elif demand_based_pricing:
                    # Get recent sales to determine demand
                    recent_sales = sum(daily_sales.get((location_id, product_id), [])[-14:])
                    if recent_sales > 100:  # High demand
                        price_change_factor = np.random.uniform(1.02, 1.06)
                    else:  # Low demand
                        price_change_factor = np.random.uniform(0.95, 0.99)
                else:
                    price_change_factor = np.random.uniform(1.01, 1.05) # Regular increase
                
                product_info_all.loc[product_id, 'current_price'] *= price_change_factor
                product_info_all.loc[product_id, 'last_price_change_date'] = date
            
            # --- ENHANCED SALES LOGIC ---
            
            # Base sales from history
            sales_history_key = (location_id, product_id)
            sales_last_7_day = sum(daily_sales.get(sales_history_key, [])[-7:])
            sales_last_14_day = sum(daily_sales.get(sales_history_key, [])[-14:])
            
            # Trend analysis
            if sales_last_14_day > 0 and sales_last_7_day > 0:
                recent_avg = sales_last_7_day / 7
                older_avg = (sales_last_14_day - sales_last_7_day) / 7
                trend_factor = min(recent_avg / max(older_avg, 1), 2.0) # Cap at 2x
            else:
                trend_factor = 1.0
            
            base_sales = sales_last_7_day if sales_last_7_day > 0 else np.random.randint(15, 45)
            
            # --- APPLY ALL FACTORS WITH INDIVIDUAL CAPS ---
            sales_multiplier = 1.0
            
            # 1. Location factors (cap at 2x)
            location_factor = min(location_info.loc[location_id, 'base_traffic'], 2.0)
            sales_multiplier *= location_factor
            if is_weekend:
                weekend_factor = min(location_info.loc[location_id, 'weekend_boost'], 1.8)
                sales_multiplier *= weekend_factor
            
            # 2. Holiday effect (cap at 2x total)
            if is_holiday:
                category = product_info_all.loc[product_id, 'product_category']
                if category == 'Beverage':
                    holiday_factor = np.random.uniform(1.2, 1.6)  # Reduced from 1.4-2.0
                else:
                    holiday_factor = np.random.uniform(1.1, 1.4)  # Reduced from 1.2-1.6
                sales_multiplier *= holiday_factor
            
            # 3. Seasonality (cap individual components)
            seasonal_factor = np.clip(product_info_all.loc[product_id, 'seasonality_factor'], 0.8, 1.3)
            month_seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * (month - 3) / 12)  # Reduced from 0.3
            seasonal_total = np.clip(seasonal_factor * month_seasonal, 0.7, 1.5)
            sales_multiplier *= seasonal_total
            
            # 4. Weather impact (cap at reasonable range)
            weather_sensitivity = np.clip(product_info_all.loc[product_id, 'weather_sensitivity'], 0.1, 0.5)  # Reduced max
            weather_impact = 1.0 + weather_sensitivity * (daily_weather - 0.5)
            weather_impact = np.clip(weather_impact, 0.8, 1.2)  # Cap weather effect
            sales_multiplier *= weather_impact
            
            # 5. Marketing campaign boost (reduced and capped)
            if is_marketing_active:
                marketing_boost = np.random.uniform(1.2, 1.4)  # Reduced from 1.3-1.8
                sales_multiplier *= marketing_boost
            
            # 6. Economic factors (cap at reasonable range)
            economic_factor = np.clip(economic_factor, 0.9, 1.1)  # Tighter range
            sales_multiplier *= economic_factor
            
            # 7. Price elasticity (cap the impact)
            if not pd.isna(product_info_all.loc[product_id, 'last_price_change_date']):
                days_since_price_change = (date - product_info_all.loc[product_id, 'last_price_change_date']).days
                if days_since_price_change < 14:
                    price_impact = max(0.7, 1.0 - 0.03 * days_since_price_change)  # Gentler decline
                    sales_multiplier *= price_impact
            
            # 8. Day-of-week patterns (reduced variance)
            dow_multipliers = {
                'Monday': 0.85, 'Tuesday': 0.9, 'Wednesday': 0.95, 'Thursday': 1.0,
                'Friday': 1.15, 'Saturday': 1.3, 'Sunday': 1.1  # Reduced from 1.4 Saturday
            }
            sales_multiplier *= dow_multipliers[day_of_week]
            
            # 9. Cross-category effects (reduced impact) - WITHIN TENANT ONLY
            if product_info_all.loc[product_id, 'product_category'] == 'Dessert':
                main_course_sales = []
                # Only look at main courses from the SAME tenant
                for pid in tenant_products:
                    if product_info_all.loc[pid, 'product_category'] == 'Main Course':
                        recent_sales = daily_sales.get((location_id, pid), [])
                        if len(recent_sales) > 0:
                            main_course_sales.append(recent_sales[-1])
                
                if len(main_course_sales) > 0:
                    avg_main_sales = np.mean(main_course_sales)
                    cross_category_boost = min(1.0 + avg_main_sales / 200, 1.25)  # Reduced from /100 and 1.5 cap
                    sales_multiplier *= cross_category_boost
            
            # 10. Apply trend (cap the trend factor)
            trend_factor = np.clip(trend_factor, 0.5, 2.0)
            sales_multiplier *= trend_factor
            
            # 11. Random stockout (reduced impact)
            is_stockout = np.random.rand() < 0.02
            if is_stockout:
                sales_multiplier *= 0.5  # Reduced from 0.3
            
            # FINAL SAFETY CAP - prevent any extreme multipliers
            sales_multiplier = np.clip(sales_multiplier, 0.1, 4.0)  # Hard cap at 4x maximum
            
            # Calculate final sales with validation
            target_sales_qty = max(1, int(base_sales * sales_multiplier * np.random.uniform(0.8, 1.2)))
            
            # Sanity check - cap at reasonable maximum
            if target_sales_qty > 1000:  # Cap daily sales at 1000 units per product
                target_sales_qty = np.random.randint(800, 1001)
            
            # Debug: Check for extreme multipliers
            if sales_multiplier > 10:
                print(f"Warning: Extreme sales multiplier {sales_multiplier:.2f} on {date} for product {product_id}")
                sales_multiplier = min(sales_multiplier, 5.0)  # Cap at 5x
            
            # Update sales history
            if sales_history_key not in daily_sales:
                daily_sales[sales_history_key] = []
            daily_sales[sales_history_key].append(target_sales_qty)
            
            # --- CALCULATE DERIVED METRICS ---
            revenue = target_sales_qty * product_info_all.loc[product_id, 'current_price']
            
            # Customer satisfaction (inversely related to stockouts and high prices)
            base_satisfaction = 4.2
            if is_stockout:
                customer_satisfaction = np.random.uniform(2.0, 3.0)
            else:
                price_factor = min(product_info_all.loc[product_id, 'current_price'] / 200, 2.0)
                customer_satisfaction = max(1.0, base_satisfaction - 0.3 * price_factor + np.random.normal(0, 0.3))
                customer_satisfaction = min(customer_satisfaction, 5.0)
            
            # --- ADD TO RECORDS ---
            records.append({
                'date': date,
                'tenant_id': tenant_id,
                'location_id': location_id,
                'product_id': product_id,
                'day_of_week': day_of_week,
                'is_holiday': is_holiday,
                'product_category': product_info_all.loc[product_id, 'product_category'],
                'current_price': product_info_all.loc[product_id, 'current_price'],
                'sales_last_7_day': sales_last_7_day,
                'target_sales_qty': target_sales_qty
            })

    # --- 3. CREATE FINAL DATAFRAME ---
    df = pd.DataFrame(records)
    
    # Data cleaning and formatting - only for the fields we're keeping
    df['date'] = pd.to_datetime(df['date'])
    df['current_price'] = df['current_price'].round(2)
    
    return df

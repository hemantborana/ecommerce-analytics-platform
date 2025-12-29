import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

# Create data folder if doesn't exist
os.makedirs('data', exist_ok=True)

# Generate date range (24 months)
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Base parameters
n_customers = 2000
n_transactions = 15000

# Customer master data
customers = pd.DataFrame({
    'customer_id': [f'C{str(i).zfill(5)}' for i in range(1, n_customers + 1)],
    'segment': np.random.choice(['Premium', 'Regular', 'Budget'], n_customers, p=[0.2, 0.5, 0.3]),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_customers, p=[0.3, 0.25, 0.25, 0.2]),
    'acquisition_date': np.random.choice(pd.date_range('2022-01-01', '2023-06-30'), n_customers),
    'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-60', '60+'], n_customers, p=[0.15, 0.35, 0.25, 0.15, 0.1])
})

# Transaction data
transactions = []

for i in range(n_transactions):
    cust = customers.sample(1).iloc[0]
    
    trans_date = np.random.choice(dates)
    trans_date = pd.Timestamp(trans_date)
    month = trans_date.month
    
    # Seasonality effect
    seasonal_factor = 1.0
    if month in [11, 12]:  # Holiday season
        seasonal_factor = 1.5
    elif month in [1, 2]:  # Post-holiday dip
        seasonal_factor = 0.7
    elif month in [6, 7]:  # Mid-year sale
        seasonal_factor = 1.2
    
    # Segment-based pricing
    base_price = {'Premium': 150, 'Regular': 80, 'Budget': 40}[cust['segment']]
    revenue = np.random.gamma(2, base_price) * seasonal_factor
    
    # Product categories
    category = np.random.choice(['Electronics', 'Fashion', 'Home', 'Beauty', 'Sports'], 
                                p=[0.25, 0.30, 0.20, 0.15, 0.10])
    
    # Channel distribution
    channel = np.random.choice(['Web', 'Mobile App', 'Store'], p=[0.45, 0.35, 0.20])
    
    # Payment method
    payment = np.random.choice(['Credit Card', 'Debit Card', 'UPI', 'Wallet'], p=[0.4, 0.25, 0.25, 0.1])
    
    # Delivery time varies by region
    delivery_days = {'North': 3, 'South': 4, 'East': 5, 'West': 3}[cust['region']]
    delivery_days += np.random.randint(-1, 3)
    
    # Customer satisfaction (correlated with delivery time)
    satisfaction = max(1, min(5, int(6 - (delivery_days / 2) + np.random.normal(0, 0.5))))
    
    transactions.append({
        'transaction_id': f'T{str(i+1).zfill(6)}',
        'customer_id': cust['customer_id'],
        'date': trans_date,
        'revenue': round(revenue, 2),
        'category': category,
        'channel': channel,
        'payment_method': payment,
        'delivery_days': delivery_days,
        'satisfaction_score': satisfaction,
        'segment': cust['segment'],
        'region': cust['region'],
        'age_group': cust['age_group']
    })

df_trans = pd.DataFrame(transactions)

# Add trend component
df_trans = df_trans.sort_values('date').reset_index(drop=True)
df_trans['month_num'] = (df_trans['date'].dt.year - 2023) * 12 + df_trans['date'].dt.month
trend_factor = 1 + (df_trans['month_num'] * 0.015)
df_trans['revenue'] = (df_trans['revenue'] * trend_factor).round(2)

# Create churn indicators (customers who haven't purchased in last 90 days)
last_purchase = df_trans.groupby('customer_id')['date'].max().reset_index()
last_purchase.columns = ['customer_id', 'last_purchase_date']
customers = customers.merge(last_purchase, on='customer_id', how='left')
customers['days_since_purchase'] = (datetime(2024, 12, 31) - customers['last_purchase_date']).dt.days
customers['is_churned'] = (customers['days_since_purchase'] > 90).astype(int)

# Save files
df_trans.to_csv('data/transactions.csv', index=False)
customers.to_csv('data/customers.csv', index=False)

print(f"Generated {len(df_trans)} transactions")
print(f"Generated {len(customers)} customers")
print(f"\nDate range: {df_trans['date'].min()} to {df_trans['date'].max()}")
print(f"\nRevenue range: ${df_trans['revenue'].min():.2f} to ${df_trans['revenue'].max():.2f}")
print(f"Total revenue: ${df_trans['revenue'].sum():,.2f}")
print(f"\nChurn rate: {customers['is_churned'].mean()*100:.1f}%")
print("\nFiles saved in 'data/' folder: transactions.csv, customers.csv")
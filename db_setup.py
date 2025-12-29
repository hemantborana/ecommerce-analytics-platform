import pandas as pd
import sqlite3
import os

# Load data
df_trans = pd.read_csv('data/transactions.csv')
df_cust = pd.read_csv('data/customers.csv')

# Convert date columns
df_trans['date'] = pd.to_datetime(df_trans['date'])
df_cust['acquisition_date'] = pd.to_datetime(df_cust['acquisition_date'])
df_cust['last_purchase_date'] = pd.to_datetime(df_cust['last_purchase_date'])

# Create database
conn = sqlite3.connect('data/analytics.db')

# Store tables
df_trans.to_sql('transactions', conn, if_exists='replace', index=False)
df_cust.to_sql('customers', conn, if_exists='replace', index=False)

# Create indexes for performance
cursor = conn.cursor()
cursor.execute('CREATE INDEX IF NOT EXISTS idx_trans_date ON transactions(date)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_trans_cust ON transactions(customer_id)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_cust_id ON customers(customer_id)')

conn.commit()
conn.close()

print("Database created: data/analytics.db")
print(f"Tables: transactions ({len(df_trans)} rows), customers ({len(df_cust)} rows)")
print("Indexes created for optimized queries")
import pandas as pd
import sqlite3
from datetime import datetime

def get_db_connection():
    return sqlite3.connect('data/analytics.db')

def load_transactions():
    conn = get_db_connection()
    df = pd.read_sql('SELECT * FROM transactions', conn)
    df['date'] = pd.to_datetime(df['date'])
    conn.close()
    return df

def load_customers():
    conn = get_db_connection()
    df = pd.read_sql('SELECT * FROM customers', conn)
    df['acquisition_date'] = pd.to_datetime(df['acquisition_date'])
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
    conn.close()
    return df

def calculate_kpis(df, period='M'):
    df_grouped = df.groupby(df['date'].dt.to_period(period)).agg({
        'transaction_id': 'count',
        'revenue': 'sum',
        'customer_id': 'nunique',
        'satisfaction_score': 'mean'
    }).reset_index()
    
    df_grouped.columns = ['period', 'total_orders', 'total_revenue', 'unique_customers', 'avg_satisfaction']
    df_grouped['aov'] = df_grouped['total_revenue'] / df_grouped['total_orders']
    df_grouped['period'] = df_grouped['period'].astype(str)
    
    return df_grouped

def get_current_kpis(df):
    total_revenue = df['revenue'].sum()
    total_orders = len(df)
    total_customers = df['customer_id'].nunique()
    avg_order_value = total_revenue / total_orders
    avg_satisfaction = df['satisfaction_score'].mean()
    
    return {
        'total_revenue': total_revenue,
        'total_orders': total_orders,
        'total_customers': total_customers,
        'aov': avg_order_value,
        'avg_satisfaction': avg_satisfaction
    }
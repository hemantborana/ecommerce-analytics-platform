import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import load_transactions, load_customers, calculate_kpis, get_current_kpis
from datetime import datetime, timedelta

st.set_page_config(page_title="Analytics Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    return load_transactions(), load_customers()

df_trans, df_cust = load_data()

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    value=(df_trans['date'].min(), df_trans['date'].max()),
    min_value=df_trans['date'].min(),
    max_value=df_trans['date'].max()
)

regions = st.sidebar.multiselect(
    "Region",
    options=df_trans['region'].unique(),
    default=df_trans['region'].unique()
)

segments = st.sidebar.multiselect(
    "Customer Segment",
    options=df_trans['segment'].unique(),
    default=df_trans['segment'].unique()
)

# Apply filters
if len(date_range) == 2:
    df_filtered = df_trans[
        (df_trans['date'] >= pd.Timestamp(date_range[0])) &
        (df_trans['date'] <= pd.Timestamp(date_range[1])) &
        (df_trans['region'].isin(regions)) &
        (df_trans['segment'].isin(segments))
    ]
else:
    df_filtered = df_trans[
        (df_trans['region'].isin(regions)) &
        (df_trans['segment'].isin(segments))
    ]

# Main dashboard
st.title("E-Commerce Analytics Dashboard")

# KPI metrics
kpis = get_current_kpis(df_filtered)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Revenue", f"${kpis['total_revenue']:,.0f}")
with col2:
    st.metric("Total Orders", f"{kpis['total_orders']:,}")
with col3:
    st.metric("Unique Customers", f"{kpis['total_customers']:,}")
with col4:
    st.metric("Avg Order Value", f"${kpis['aov']:.2f}")
with col5:
    st.metric("Avg Satisfaction", f"{kpis['avg_satisfaction']:.2f}/5")

st.divider()

# Revenue trend
st.subheader("Revenue Trend Over Time")
monthly_data = calculate_kpis(df_filtered, 'M')

fig_revenue = px.line(
    monthly_data,
    x='period',
    y='total_revenue',
    title='Monthly Revenue',
    labels={'period': 'Month', 'total_revenue': 'Revenue ($)'}
)
fig_revenue.update_traces(line_color='#1f77b4', line_width=3)
st.plotly_chart(fig_revenue, use_container_width=True)

# Two column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue by Region")
    region_data = df_filtered.groupby('region')['revenue'].sum().reset_index()
    fig_region = px.bar(
        region_data,
        x='region',
        y='revenue',
        color='region',
        labels={'revenue': 'Revenue ($)', 'region': 'Region'}
    )
    st.plotly_chart(fig_region, use_container_width=True)

with col2:
    st.subheader("Revenue by Segment")
    segment_data = df_filtered.groupby('segment')['revenue'].sum().reset_index()
    fig_segment = px.pie(
        segment_data,
        values='revenue',
        names='segment',
        hole=0.4
    )
    st.plotly_chart(fig_segment, use_container_width=True)

# Category performance
st.subheader("Product Category Performance")
category_data = df_filtered.groupby('category').agg({
    'revenue': 'sum',
    'transaction_id': 'count',
    'satisfaction_score': 'mean'
}).reset_index()
category_data.columns = ['Category', 'Revenue', 'Orders', 'Avg Satisfaction']
category_data = category_data.sort_values('Revenue', ascending=False)

fig_category = px.bar(
    category_data,
    x='Category',
    y='Revenue',
    color='Avg Satisfaction',
    color_continuous_scale='RdYlGn',
    labels={'Revenue': 'Revenue ($)'}
)
st.plotly_chart(fig_category, use_container_width=True)

# Channel distribution
st.subheader("Sales Channel Distribution")
col1, col2 = st.columns(2)

with col1:
    channel_data = df_filtered.groupby('channel')['revenue'].sum().reset_index()
    fig_channel = px.bar(
        channel_data,
        x='channel',
        y='revenue',
        color='channel',
        labels={'revenue': 'Revenue ($)', 'channel': 'Channel'}
    )
    st.plotly_chart(fig_channel, use_container_width=True)

with col2:
    age_data = df_filtered.groupby('age_group')['revenue'].sum().reset_index()
    fig_age = px.bar(
        age_data,
        x='age_group',
        y='revenue',
        color='age_group',
        labels={'revenue': 'Revenue ($)', 'age_group': 'Age Group'}
    )
    st.plotly_chart(fig_age, use_container_width=True)
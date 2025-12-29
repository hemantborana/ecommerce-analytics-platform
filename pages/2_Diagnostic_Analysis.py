import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import load_transactions, load_customers
import numpy as np

st.set_page_config(page_title="Diagnostic Analysis", layout="wide")

@st.cache_data
def load_data():
    return load_transactions(), load_customers()

df_trans, df_cust = load_data()

st.title("Diagnostic Analysis & Root Cause Investigation")

tab1, tab2, tab3 = st.tabs(["Drill-Down Analysis", "Root Cause Analysis", "Correlation Analysis"])

# Tab 1: Drill-Down Analysis
with tab1:
    st.header("Multi-Dimensional Drill-Down")
    
    drill_dimension = st.selectbox(
        "Select Drill-Down Dimension",
        ["Geographic", "Time-Based", "Product Category", "Customer Segment"]
    )
    
    if drill_dimension == "Geographic":
        st.subheader("Geographic Analysis: Region → City Level")
        
        region_revenue = df_trans.groupby('region').agg({
            'revenue': 'sum',
            'transaction_id': 'count',
            'satisfaction_score': 'mean'
        }).reset_index()
        region_revenue.columns = ['Region', 'Revenue', 'Orders', 'Avg Satisfaction']
        region_revenue = region_revenue.sort_values('Revenue', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                region_revenue,
                x='Region',
                y='Revenue',
                color='Avg Satisfaction',
                color_continuous_scale='RdYlGn',
                title='Revenue by Region'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(region_revenue.style.format({
                'Revenue': '${:,.0f}',
                'Orders': '{:,}',
                'Avg Satisfaction': '{:.2f}'
            }), use_container_width=True)
        
        selected_region = st.selectbox("Drill into Region", region_revenue['Region'].tolist())
        
        if selected_region:
            region_data = df_trans[df_trans['region'] == selected_region]
            
            segment_breakdown = region_data.groupby('segment').agg({
                'revenue': 'sum',
                'transaction_id': 'count'
            }).reset_index()
            segment_breakdown.columns = ['Segment', 'Revenue', 'Orders']
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(
                    segment_breakdown,
                    values='Revenue',
                    names='Segment',
                    title=f'{selected_region} - Revenue by Segment'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                category_breakdown = region_data.groupby('category')['revenue'].sum().reset_index()
                fig = px.bar(
                    category_breakdown,
                    x='category',
                    y='revenue',
                    title=f'{selected_region} - Revenue by Category'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif drill_dimension == "Time-Based":
        st.subheader("Time Analysis: Year → Quarter → Month → Day")
        
        df_trans['year'] = df_trans['date'].dt.year
        df_trans['quarter'] = df_trans['date'].dt.quarter
        df_trans['month'] = df_trans['date'].dt.month
        
        year_revenue = df_trans.groupby('year')['revenue'].sum().reset_index()
        
        fig = px.bar(year_revenue, x='year', y='revenue', title='Yearly Revenue')
        st.plotly_chart(fig, use_container_width=True)
        
        selected_year = st.selectbox("Select Year", sorted(df_trans['year'].unique()))
        
        year_data = df_trans[df_trans['year'] == selected_year]
        quarter_revenue = year_data.groupby('quarter')['revenue'].sum().reset_index()
        quarter_revenue['quarter'] = 'Q' + quarter_revenue['quarter'].astype(str)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(quarter_revenue, x='quarter', y='revenue', title=f'{selected_year} - Quarterly Revenue')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            selected_quarter = st.selectbox("Select Quarter", quarter_revenue['quarter'].tolist())
            q_num = int(selected_quarter[1])
            quarter_data = year_data[year_data['quarter'] == q_num]
            
            month_revenue = quarter_data.groupby('month')['revenue'].sum().reset_index()
            fig = px.line(month_revenue, x='month', y='revenue', title=f'{selected_year} {selected_quarter} - Monthly Revenue')
            st.plotly_chart(fig, use_container_width=True)
    
    elif drill_dimension == "Product Category":
        st.subheader("Product Analysis: Category → Subcategory")
        
        category_revenue = df_trans.groupby('category').agg({
            'revenue': 'sum',
            'transaction_id': 'count',
            'satisfaction_score': 'mean'
        }).reset_index()
        category_revenue.columns = ['Category', 'Revenue', 'Orders', 'Avg Satisfaction']
        category_revenue = category_revenue.sort_values('Revenue', ascending=False)
        
        fig = px.bar(
            category_revenue,
            x='Category',
            y='Revenue',
            color='Avg Satisfaction',
            color_continuous_scale='RdYlGn',
            title='Revenue by Category'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        selected_category = st.selectbox("Select Category", category_revenue['Category'].tolist())
        
        cat_data = df_trans[df_trans['category'] == selected_category]
        
        col1, col2 = st.columns(2)
        
        with col1:
            channel_breakdown = cat_data.groupby('channel')['revenue'].sum().reset_index()
            fig = px.pie(channel_breakdown, values='revenue', names='channel', 
                        title=f'{selected_category} - Revenue by Channel')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            segment_breakdown = cat_data.groupby('segment')['revenue'].sum().reset_index()
            fig = px.bar(segment_breakdown, x='segment', y='revenue',
                        title=f'{selected_category} - Revenue by Segment')
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Customer Segment
        st.subheader("Customer Segment Analysis")
        
        segment_revenue = df_trans.groupby('segment').agg({
            'revenue': 'sum',
            'transaction_id': 'count',
            'customer_id': 'nunique',
            'satisfaction_score': 'mean'
        }).reset_index()
        segment_revenue.columns = ['Segment', 'Revenue', 'Orders', 'Customers', 'Avg Satisfaction']
        segment_revenue['Revenue per Customer'] = segment_revenue['Revenue'] / segment_revenue['Customers']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                segment_revenue,
                x='Segment',
                y='Revenue',
                color='Avg Satisfaction',
                color_continuous_scale='RdYlGn',
                title='Revenue by Customer Segment'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(segment_revenue.style.format({
                'Revenue': '${:,.0f}',
                'Orders': '{:,}',
                'Customers': '{:,}',
                'Avg Satisfaction': '{:.2f}',
                'Revenue per Customer': '${:,.2f}'
            }), use_container_width=True)
        
        selected_segment = st.selectbox("Select Segment", segment_revenue['Segment'].tolist())
        
        seg_data = df_trans[df_trans['segment'] == selected_segment]
        
        col1, col2 = st.columns(2)
        
        with col1:
            age_breakdown = seg_data.groupby('age_group')['revenue'].sum().reset_index()
            fig = px.bar(age_breakdown, x='age_group', y='revenue',
                        title=f'{selected_segment} - Revenue by Age Group')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            region_breakdown = seg_data.groupby('region')['revenue'].sum().reset_index()
            fig = px.pie(region_breakdown, values='revenue', names='region',
                        title=f'{selected_segment} - Revenue by Region')
            st.plotly_chart(fig, use_container_width=True)

# Tab 2: Root Cause Analysis
with tab2:
    st.header("Root Cause Analysis")
    
    analysis_type = st.selectbox(
        "Select Issue to Investigate",
        ["Revenue Decline", "Low Satisfaction", "High Delivery Time", "Customer Churn"]
    )
    
    if analysis_type == "Revenue Decline":
        st.subheader("Revenue Decline Investigation")
        
        df_trans['month_year'] = df_trans['date'].dt.to_period('M')
        monthly_rev = df_trans.groupby('month_year')['revenue'].sum().reset_index()
        monthly_rev['month_year'] = monthly_rev['month_year'].astype(str)
        monthly_rev['pct_change'] = monthly_rev['revenue'].pct_change() * 100
        
        declining_months = monthly_rev[monthly_rev['pct_change'] < -5]
        
        if len(declining_months) > 0:
            st.warning(f"Found {len(declining_months)} months with >5% revenue decline")
            
            fig = px.line(monthly_rev, x='month_year', y='revenue', title='Monthly Revenue Trend')
            fig.add_hline(y=monthly_rev['revenue'].mean(), line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            
            worst_month = declining_months.loc[declining_months['pct_change'].idxmin(), 'month_year']
            st.error(f"Worst decline: {worst_month}")
            
            month_data = df_trans[df_trans['month_year'] == worst_month]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Contributing Factors:**")
                region_impact = month_data.groupby('region')['revenue'].sum().sort_values()
                st.write(f"- Lowest performing region: {region_impact.index[0]} (${region_impact.values[0]:,.0f})")
                
                category_impact = month_data.groupby('category')['revenue'].sum().sort_values()
                st.write(f"- Lowest performing category: {category_impact.index[0]} (${category_impact.values[0]:,.0f})")
                
                avg_sat = month_data['satisfaction_score'].mean()
                st.write(f"- Average satisfaction: {avg_sat:.2f}")
            
            with col2:
                fig = px.bar(region_impact, title='Revenue by Region in Decline Month')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No significant revenue declines detected")
    
    elif analysis_type == "Low Satisfaction":
        st.subheader("Low Satisfaction Investigation")
        
        avg_satisfaction = df_trans['satisfaction_score'].mean()
        low_sat = df_trans[df_trans['satisfaction_score'] <= 2]
        
        st.metric("Overall Avg Satisfaction", f"{avg_satisfaction:.2f}/5")
        st.metric("Low Satisfaction Orders", f"{len(low_sat)} ({len(low_sat)/len(df_trans)*100:.1f}%)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            region_sat = df_trans.groupby('region')['satisfaction_score'].mean().sort_values()
            fig = px.bar(region_sat, title='Avg Satisfaction by Region')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            category_sat = df_trans.groupby('category')['satisfaction_score'].mean().sort_values()
            fig = px.bar(category_sat, title='Avg Satisfaction by Category')
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Key Findings:**")
        worst_region = region_sat.index[0]
        st.write(f"- Region with lowest satisfaction: {worst_region} ({region_sat.values[0]:.2f})")
        
        worst_region_delivery = df_trans[df_trans['region'] == worst_region]['delivery_days'].mean()
        overall_delivery = df_trans['delivery_days'].mean()
        st.write(f"- Avg delivery time in {worst_region}: {worst_region_delivery:.1f} days (Overall: {overall_delivery:.1f} days)")
    
    elif analysis_type == "High Delivery Time":
        st.subheader("Delivery Time Analysis")
        
        avg_delivery = df_trans['delivery_days'].mean()
        st.metric("Average Delivery Time", f"{avg_delivery:.1f} days")
        
        region_delivery = df_trans.groupby('region').agg({
            'delivery_days': 'mean',
            'satisfaction_score': 'mean'
        }).reset_index()
        
        fig = px.scatter(
            region_delivery,
            x='delivery_days',
            y='satisfaction_score',
            size=[50]*len(region_delivery),
            text='region',
            title='Delivery Time vs Satisfaction by Region'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        correlation = df_trans[['delivery_days', 'satisfaction_score']].corr().iloc[0, 1]
        st.metric("Correlation: Delivery Time vs Satisfaction", f"{correlation:.3f}")
        
        if correlation < -0.3:
            st.warning("Strong negative correlation: Longer delivery times reduce satisfaction")
    
    else:  # Customer Churn
        st.subheader("Customer Churn Analysis")
        
        churn_rate = df_cust['is_churned'].mean() * 100
        st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
        
        churned = df_cust[df_cust['is_churned'] == 1]
        active = df_cust[df_cust['is_churned'] == 0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            churn_by_segment = df_cust.groupby('segment')['is_churned'].mean() * 100
            fig = px.bar(churn_by_segment, title='Churn Rate by Segment (%)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            churn_by_region = df_cust.groupby('region')['is_churned'].mean() * 100
            fig = px.bar(churn_by_region, title='Churn Rate by Region (%)')
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Churn Characteristics:**")
        st.write(f"- Highest churn segment: {churn_by_segment.idxmax()} ({churn_by_segment.max():.1f}%)")
        st.write(f"- Highest churn region: {churn_by_region.idxmax()} ({churn_by_region.max():.1f}%)")

# Tab 3: Correlation Analysis
with tab3:
    st.header("Correlation Analysis")
    
    numeric_cols = ['revenue', 'delivery_days', 'satisfaction_score']
    corr_df = df_trans[numeric_cols].corr()
    
    fig = px.imshow(
        corr_df,
        text_auto='.2f',
        color_continuous_scale='RdBu',
        title='Correlation Matrix'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Key Correlations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            df_trans,
            x='delivery_days',
            y='satisfaction_score',
            trendline='ols',
            title='Delivery Days vs Satisfaction Score'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df_trans,
            x='revenue',
            y='satisfaction_score',
            trendline='ols',
            title='Revenue vs Satisfaction Score'
        )
        st.plotly_chart(fig, use_container_width=True)
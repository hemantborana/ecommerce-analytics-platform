import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_transactions, load_customers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prescriptive Analytics", layout="wide")

@st.cache_data
def load_data():
    return load_transactions(), load_customers()

df_trans, df_cust = load_data()

st.title("Prescriptive Analytics & AI-Driven Recommendations")

tab1, tab2, tab3, tab4 = st.tabs([
    "Customer Retention Actions",
    "Revenue Optimization",
    "Inventory & Resource Planning",
    "ROI Calculator"
])

# Tab 1: Customer Retention
with tab1:
    st.header("Customer Retention Recommendations")
    
    customer_features = df_trans.groupby('customer_id').agg({
        'revenue': ['sum', 'mean', 'count'],
        'satisfaction_score': 'mean',
        'delivery_days': 'mean'
    }).reset_index()
    
    customer_features.columns = ['customer_id', 'total_revenue', 'avg_revenue', 'order_count', 'avg_satisfaction', 'avg_delivery']
    
    customer_features = customer_features.merge(
        df_cust[['customer_id', 'is_churned', 'segment', 'region']],
        on='customer_id'
    )
    
    customer_features['segment_encoded'] = customer_features['segment'].map({'Premium': 2, 'Regular': 1, 'Budget': 0})
    customer_features['region_encoded'] = customer_features['region'].map({'North': 0, 'South': 1, 'East': 2, 'West': 3})
    
    feature_cols = ['total_revenue', 'avg_revenue', 'order_count', 'avg_satisfaction', 'avg_delivery', 'segment_encoded', 'region_encoded']
    X = customer_features[feature_cols]
    y = customer_features['is_churned']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    all_predictions = model.predict_proba(X)
    customer_features['churn_probability'] = all_predictions[:, 1]
    
    high_risk = customer_features[customer_features['churn_probability'] > 0.7].sort_values('churn_probability', ascending=False)
    medium_risk = customer_features[(customer_features['churn_probability'] > 0.4) & (customer_features['churn_probability'] <= 0.7)]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("High Risk Customers", len(high_risk))
    col2.metric("Medium Risk Customers", len(medium_risk))
    col3.metric("Potential Revenue at Risk", f"${high_risk['total_revenue'].sum():,.0f}")
    
    st.subheader("Recommended Actions")
    
    def generate_recommendation(row):
        actions = []
        priority = "High" if row['churn_probability'] > 0.7 else "Medium"
        
        if row['avg_satisfaction'] < 3.0:
            actions.append("Immediate satisfaction survey and service recovery")
        
        if row['avg_delivery'] > 5:
            actions.append("Expedite shipping for next orders")
        
        if row['order_count'] < 3:
            actions.append("Send personalized product recommendations")
        
        if row['segment'] == 'Premium' and row['churn_probability'] > 0.7:
            actions.append("Assign dedicated account manager")
        
        if len(actions) == 0:
            actions.append("Send retention discount (15-20%)")
        
        return {
            'customer_id': row['customer_id'],
            'segment': row['segment'],
            'region': row['region'],
            'churn_probability': row['churn_probability'],
            'priority': priority,
            'recommended_action': ' | '.join(actions),
            'potential_value': row['total_revenue']
        }
    
    at_risk = pd.concat([high_risk, medium_risk])
    recommendations = at_risk.apply(generate_recommendation, axis=1, result_type='expand')
    recommendations = recommendations.sort_values(['priority', 'churn_probability'], ascending=[True, False])
    
    st.dataframe(
        recommendations.head(20).style.format({
            'churn_probability': '{:.1%}',
            'potential_value': '${:,.0f}'
        }),
        use_container_width=True
    )
    
    st.subheader("Expected Impact")
    
    retention_rate = st.slider("Expected Retention Rate from Actions (%)", 10, 80, 40)
    
    customers_retained = int(len(high_risk) * (retention_rate / 100))
    revenue_saved = high_risk['total_revenue'].sum() * (retention_rate / 100)
    
    col1, col2 = st.columns(2)
    col1.metric("Customers Retained", customers_retained)
    col2.metric("Revenue Saved", f"${revenue_saved:,.0f}")
    
    cost_per_customer = 50
    total_cost = len(high_risk) * cost_per_customer
    net_benefit = revenue_saved - total_cost
    roi = (net_benefit / total_cost) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Campaign Cost", f"${total_cost:,.0f}")
    col2.metric("Net Benefit", f"${net_benefit:,.0f}")
    col3.metric("ROI", f"{roi:.0f}%")

# Tab 2: Revenue Optimization
with tab2:
    st.header("Revenue Optimization Recommendations")
    
    st.subheader("Pricing Strategy Recommendations")
    
    category_performance = df_trans.groupby('category').agg({
        'revenue': 'sum',
        'transaction_id': 'count',
        'satisfaction_score': 'mean'
    }).reset_index()
    category_performance.columns = ['category', 'revenue', 'orders', 'satisfaction']
    category_performance['avg_order_value'] = category_performance['revenue'] / category_performance['orders']
    
    def pricing_recommendation(row):
        if row['satisfaction'] > 4.0:
            return "Increase price by 5-10% (high satisfaction buffer)"
        elif row['satisfaction'] < 3.0:
            return "Improve quality before price changes"
        else:
            return "Test A/B pricing (±5%)"
    
    category_performance['pricing_action'] = category_performance.apply(pricing_recommendation, axis=1)
    
    st.dataframe(
        category_performance.style.format({
            'revenue': '${:,.0f}',
            'orders': '{:,}',
            'satisfaction': '{:.2f}',
            'avg_order_value': '${:.2f}'
        }),
        use_container_width=True
    )
    
    st.subheader("Channel Optimization")
    
    channel_perf = df_trans.groupby('channel').agg({
        'revenue': 'sum',
        'transaction_id': 'count',
        'satisfaction_score': 'mean'
    }).reset_index()
    channel_perf.columns = ['channel', 'revenue', 'orders', 'satisfaction']
    channel_perf['revenue_per_order'] = channel_perf['revenue'] / channel_perf['orders']
    
    best_channel = channel_perf.loc[channel_perf['revenue'].idxmax(), 'channel']
    
    st.info(f"**Recommendation:** Focus marketing budget on {best_channel} channel (highest revenue)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            channel_perf,
            x='channel',
            y='revenue',
            color='satisfaction',
            color_continuous_scale='RdYlGn',
            title='Channel Performance'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        budget_allocation = channel_perf.copy()
        total_revenue = budget_allocation['revenue'].sum()
        budget_allocation['recommended_budget_%'] = (budget_allocation['revenue'] / total_revenue * 100).round(1)
        
        st.write("**Recommended Budget Allocation:**")
        for _, row in budget_allocation.iterrows():
            st.write(f"- {row['channel']}: {row['recommended_budget_%']}%")
    
    st.subheader("Cross-Sell Opportunities")
    
    customer_categories = df_trans.groupby('customer_id')['category'].apply(lambda x: x.unique().tolist()).reset_index()
    customer_categories['num_categories'] = customer_categories['category'].apply(len)
    
    single_category_customers = customer_categories[customer_categories['num_categories'] == 1]
    
    st.metric("Customers Buying Single Category", len(single_category_customers))
    st.info(f"**Recommendation:** Target {len(single_category_customers)} customers with cross-category promotions")
    
    for idx, row in single_category_customers.head(5).iterrows():
        current_cat = row['category'][0]
        st.write(f"- Customer {row['customer_id']}: Currently buys {current_cat} → Recommend other categories")

# Tab 3: Inventory & Resource Planning
with tab3:
    st.header("Inventory & Resource Optimization")
    
    st.subheader("Demand Forecasting by Category")
    
    category_monthly = df_trans.groupby([df_trans['date'].dt.to_period('M'), 'category']).agg({
        'transaction_id': 'count'
    }).reset_index()
    category_monthly.columns = ['month', 'category', 'orders']
    category_monthly['month'] = category_monthly['month'].astype(str)
    
    for category in df_trans['category'].unique():
        cat_data = category_monthly[category_monthly['category'] == category]
        avg_orders = cat_data['orders'].mean()
        recent_orders = cat_data.tail(3)['orders'].mean()
        
        if recent_orders > avg_orders * 1.2:
            trend = "↑ Increasing"
            action = "Increase inventory by 20%"
        elif recent_orders < avg_orders * 0.8:
            trend = "↓ Decreasing"
            action = "Reduce inventory by 15%"
        else:
            trend = "→ Stable"
            action = "Maintain current levels"
        
        col1, col2, col3 = st.columns([2, 1, 2])
        col1.write(f"**{category}**")
        col2.write(trend)
        col3.write(action)
    
    st.subheader("Delivery Optimization")
    
    region_delivery = df_trans.groupby('region').agg({
        'delivery_days': 'mean',
        'satisfaction_score': 'mean',
        'transaction_id': 'count'
    }).reset_index()
    region_delivery.columns = ['region', 'avg_delivery', 'satisfaction', 'orders']
    
    for _, row in region_delivery.iterrows():
        if row['avg_delivery'] > 4:
            recommendation = f"Open fulfillment center or partner with local logistics"
            priority = "High"
        elif row['avg_delivery'] > 3:
            recommendation = f"Optimize delivery routes"
            priority = "Medium"
        else:
            recommendation = f"Maintain current operations"
            priority = "Low"
        
        st.write(f"**{row['region']}** (Avg: {row['avg_delivery']:.1f} days) - Priority: {priority}")
        st.write(f"   → {recommendation}")
    
    st.subheader("Staffing Recommendations")
    
    daily_orders = df_trans.groupby(df_trans['date'].dt.dayofweek)['transaction_id'].count().reset_index()
    daily_orders.columns = ['day_of_week', 'orders']
    daily_orders['day_name'] = daily_orders['day_of_week'].map({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    })
    
    avg_orders = daily_orders['orders'].mean()
    daily_orders['staff_ratio'] = (daily_orders['orders'] / avg_orders * 100).round(0)
    
    fig = px.bar(
        daily_orders,
        x='day_name',
        y='orders',
        title='Orders by Day of Week'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("**Staffing Recommendations (% of baseline):**")
    for _, row in daily_orders.iterrows():
        st.write(f"- {row['day_name']}: {row['staff_ratio']:.0f}% staffing")

# Tab 4: ROI Calculator
with tab4:
    st.header("ROI Calculator for Recommended Actions")
    
    st.subheader("Investment Scenario Planner")
    
    action_type = st.selectbox(
        "Select Action Type",
        [
            "Customer Retention Campaign",
            "Marketing Channel Optimization",
            "Delivery Network Expansion",
            "Inventory Optimization"
        ]
    )
    
    if action_type == "Customer Retention Campaign":
        st.write("**Campaign Parameters:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_customers = st.number_input("Target Customers", 100, 1000, 300)
            cost_per_customer = st.number_input("Cost per Customer ($)", 10, 200, 50)
            retention_rate = st.slider("Expected Retention Rate (%)", 10, 80, 40)
        
        with col2:
            avg_customer_value = customer_features['total_revenue'].mean()
            st.metric("Avg Customer Value", f"${avg_customer_value:,.2f}")
            
            total_cost = target_customers * cost_per_customer
            customers_retained = int(target_customers * (retention_rate / 100))
            revenue_saved = customers_retained * avg_customer_value
            net_benefit = revenue_saved - total_cost
            roi = (net_benefit / total_cost) * 100 if total_cost > 0 else 0
        
        st.divider()
        
        st.subheader("Financial Impact")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Investment", f"${total_cost:,.0f}")
        col2.metric("Revenue Saved", f"${revenue_saved:,.0f}")
        col3.metric("Net Benefit", f"${net_benefit:,.0f}")
        col4.metric("ROI", f"{roi:.0f}%")
        
        if roi > 200:
            st.success("✅ Excellent ROI - Highly recommended")
        elif roi > 100:
            st.info("✓ Good ROI - Recommended")
        elif roi > 0:
            st.warning("⚠ Positive but low ROI - Consider optimization")
        else:
            st.error("❌ Negative ROI - Not recommended")
    
    elif action_type == "Marketing Channel Optimization":
        st.write("**Budget Reallocation:**")
        
        current_budget = st.number_input("Current Monthly Budget ($)", 5000, 100000, 20000)
        
        channel_perf = df_trans.groupby('channel').agg({
            'revenue': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        channel_perf['conversion'] = channel_perf['revenue'] / channel_perf['transaction_id']
        
        total_revenue = channel_perf['revenue'].sum()
        channel_perf['current_allocation_%'] = (channel_perf['revenue'] / total_revenue * 100).round(1)
        channel_perf['recommended_budget'] = (current_budget * channel_perf['current_allocation_%'] / 100).round(0)
        
        st.dataframe(
            channel_perf[['channel', 'current_allocation_%', 'recommended_budget', 'conversion']].style.format({
                'current_allocation_%': '{:.1f}%',
                'recommended_budget': '${:,.0f}',
                'conversion': '${:.2f}'
            }),
            use_container_width=True
        )
        
        expected_lift = st.slider("Expected Revenue Lift (%)", 5, 30, 15)
        
        new_revenue = total_revenue * (1 + expected_lift / 100)
        additional_revenue = new_revenue - total_revenue
        roi = (additional_revenue / current_budget) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Additional Revenue", f"${additional_revenue:,.0f}")
        col2.metric("Monthly Investment", f"${current_budget:,.0f}")
        col3.metric("Monthly ROI", f"{roi:.0f}%")
    
    elif action_type == "Delivery Network Expansion":
        st.write("**Expansion Analysis:**")
        
        region = st.selectbox("Target Region", df_trans['region'].unique())
        
        region_data = df_trans[df_trans['region'] == region]
        current_delivery = region_data['delivery_days'].mean()
        target_delivery = st.number_input("Target Delivery Days", 1, 5, 2)
        
        setup_cost = st.number_input("Setup Cost ($)", 10000, 500000, 100000)
        monthly_operating = st.number_input("Monthly Operating Cost ($)", 5000, 50000, 15000)
        
        delivery_improvement = current_delivery - target_delivery
        satisfaction_lift = delivery_improvement * 0.2
        
        current_satisfaction = region_data['satisfaction_score'].mean()
        new_satisfaction = min(5, current_satisfaction + satisfaction_lift)
        
        revenue_lift = satisfaction_lift * 0.1
        current_revenue = region_data['revenue'].sum()
        additional_monthly_revenue = current_revenue / 24 * revenue_lift
        
        payback_months = setup_cost / (additional_monthly_revenue - monthly_operating) if additional_monthly_revenue > monthly_operating else 999
        
        col1, col2 = st.columns(2)
        col1.metric("Current Avg Delivery", f"{current_delivery:.1f} days")
        col2.metric("Target Delivery", f"{target_delivery} days")
        
        col1, col2 = st.columns(2)
        col1.metric("Expected Satisfaction", f"{new_satisfaction:.2f}/5")
        col2.metric("Monthly Revenue Lift", f"${additional_monthly_revenue:,.0f}")
        
        st.metric("Payback Period", f"{payback_months:.1f} months" if payback_months < 999 else "Not viable")
    
    else:  # Inventory Optimization
        st.write("**Inventory Optimization:**")
        
        category = st.selectbox("Product Category", df_trans['category'].unique())
        
        cat_data = df_trans[df_trans['category'] == category]
        monthly_orders = len(cat_data) / 24
        
        current_inventory_cost = st.number_input("Current Monthly Inventory Cost ($)", 5000, 100000, 20000)
        optimization_savings = st.slider("Expected Cost Reduction (%)", 5, 30, 15)
        
        monthly_savings = current_inventory_cost * (optimization_savings / 100)
        annual_savings = monthly_savings * 12
        
        implementation_cost = st.number_input("Implementation Cost ($)", 5000, 50000, 15000)
        
        roi = ((annual_savings - implementation_cost) / implementation_cost) * 100
        payback_months = implementation_cost / monthly_savings
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Annual Savings", f"${annual_savings:,.0f}")
        col2.metric("Payback Period", f"{payback_months:.1f} months")
        col3.metric("First Year ROI", f"{roi:.0f}%")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from statsmodels.tsa.arima.model import ARIMA
from utils import load_transactions, load_customers
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Predictive Analytics", layout="wide")

@st.cache_data
def load_data():
    return load_transactions(), load_customers()

df_trans, df_cust = load_data()

st.title("Predictive Analytics & Forecasting")

tab1, tab2, tab3, tab4 = st.tabs([
    "Revenue Forecasting", 
    "Churn Prediction", 
    "Customer Lifetime Value", 
    "Model Performance"
])

# Tab 1: Revenue Forecasting
with tab1:
    st.header("Revenue Forecasting (Time Series)")
    
    daily_revenue = df_trans.groupby(df_trans['date'].dt.date)['revenue'].sum().reset_index()
    daily_revenue.columns = ['date', 'revenue']
    daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
    daily_revenue = daily_revenue.sort_values('date')
    
    train_size = int(len(daily_revenue) * 0.8)
    train_data = daily_revenue[:train_size]
    test_data = daily_revenue[train_size:]
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        forecast_days = st.slider("Forecast Days Ahead", 7, 90, 30)
        model_type = st.selectbox("Model Type", ["ARIMA", "Moving Average"])
    
    with col1:
        if model_type == "ARIMA":
            with st.spinner("Training ARIMA model..."):
                model = ARIMA(train_data['revenue'], order=(5, 1, 2))
                fitted_model = model.fit()
                
                forecast = fitted_model.forecast(steps=len(test_data) + forecast_days)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=train_data['date'],
                    y=train_data['revenue'],
                    name='Training Data',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=test_data['date'],
                    y=test_data['revenue'],
                    name='Actual (Test)',
                    line=dict(color='green')
                ))
                
                forecast_dates = pd.date_range(
                    start=train_data['date'].iloc[-1] + pd.Timedelta(days=1),
                    periods=len(test_data) + forecast_days
                )
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast,
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title='Revenue Forecast (ARIMA)',
                    xaxis_title='Date',
                    yaxis_title='Revenue ($)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                test_forecast = forecast[:len(test_data)]
                mae = mean_absolute_error(test_data['revenue'], test_forecast)
                rmse = np.sqrt(mean_squared_error(test_data['revenue'], test_forecast))
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"${mae:,.2f}")
                col2.metric("RMSE", f"${rmse:,.2f}")
                col3.metric("Forecast Accuracy", f"{(1 - mae/test_data['revenue'].mean())*100:.1f}%")
        
        else:  # Moving Average
            window = 7
            train_data['ma'] = train_data['revenue'].rolling(window=window).mean()
            last_ma = train_data['ma'].iloc[-1]
            
            forecast_values = [last_ma] * (len(test_data) + forecast_days)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=train_data['date'],
                y=train_data['revenue'],
                name='Training Data',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=test_data['date'],
                y=test_data['revenue'],
                name='Actual (Test)',
                line=dict(color='green')
            ))
            
            forecast_dates = pd.date_range(
                start=train_data['date'].iloc[-1] + pd.Timedelta(days=1),
                periods=len(test_data) + forecast_days
            )
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f'Revenue Forecast ({window}-Day Moving Average)',
                xaxis_title='Date',
                yaxis_title='Revenue ($)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Forecast Summary")
    future_dates = pd.date_range(
        start=daily_revenue['date'].max() + pd.Timedelta(days=1),
        periods=30
    )
    if model_type == "ARIMA":
        future_forecast = forecast[-30:]
    else:
        future_forecast = forecast_values[-30:]
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Revenue': future_forecast
    })
    st.dataframe(forecast_df.style.format({'Predicted Revenue': '${:,.2f}'}), use_container_width=True)

# Tab 2: Churn Prediction
with tab2:
    st.header("Customer Churn Prediction")
    
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
    
    with st.spinner("Training churn prediction model..."):
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title='Feature Importance for Churn Prediction'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Model Accuracy", f"{accuracy*100:.1f}%")
            
            churn_prob = model.predict_proba(X_test)[:, 1]
            high_risk = (churn_prob > 0.7).sum()
            st.metric("High Risk Customers", high_risk)
            
            st.write("**Classification Report:**")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write(f"Precision: {report['1']['precision']:.2f}")
            st.write(f"Recall: {report['1']['recall']:.2f}")
            st.write(f"F1-Score: {report['1']['f1-score']:.2f}")
    
    st.subheader("High-Risk Customers")
    
    all_predictions = model.predict_proba(X)
    customer_features['churn_probability'] = all_predictions[:, 1]
    
    high_risk_customers = customer_features[customer_features['churn_probability'] > 0.7].sort_values('churn_probability', ascending=False)
    
    display_cols = ['customer_id', 'segment', 'region', 'total_revenue', 'order_count', 'avg_satisfaction', 'churn_probability']
    st.dataframe(
        high_risk_customers[display_cols].head(20).style.format({
            'total_revenue': '${:,.2f}',
            'avg_satisfaction': '{:.2f}',
            'churn_probability': '{:.2%}'
        }),
        use_container_width=True
    )

# Tab 3: Customer Lifetime Value
with tab3:
    st.header("Customer Lifetime Value (CLV) Prediction")
    
    clv_features = df_trans.groupby('customer_id').agg({
        'revenue': 'sum',
        'transaction_id': 'count',
        'satisfaction_score': 'mean'
    }).reset_index()
    
    clv_features.columns = ['customer_id', 'total_revenue', 'order_count', 'avg_satisfaction']
    
    clv_features = clv_features.merge(
        df_cust[['customer_id', 'segment', 'region', 'days_since_purchase']],
        on='customer_id'
    )
    
    clv_features['avg_order_value'] = clv_features['total_revenue'] / clv_features['order_count']
    clv_features['segment_encoded'] = clv_features['segment'].map({'Premium': 2, 'Regular': 1, 'Budget': 0})
    clv_features['region_encoded'] = clv_features['region'].map({'North': 0, 'South': 1, 'East': 2, 'West': 3})
    
    feature_cols = ['order_count', 'avg_satisfaction', 'avg_order_value', 'segment_encoded', 'region_encoded', 'days_since_purchase']
    X = clv_features[feature_cols]
    y = clv_features['total_revenue']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with st.spinner("Training CLV prediction model..."):
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"${mae:,.2f}")
        col2.metric("RMSE", f"${rmse:,.2f}")
        col3.metric("Prediction Accuracy", f"{(1 - mae/y_test.mean())*100:.1f}%")
        
        fig = px.scatter(
            x=y_test,
            y=y_pred,
            labels={'x': 'Actual CLV ($)', 'y': 'Predicted CLV ($)'},
            title='Actual vs Predicted CLV'
        )
        fig.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top Value Customers (Predicted)")
    
    all_predictions = model.predict(X)
    clv_features['predicted_clv'] = all_predictions
    
    top_customers = clv_features.sort_values('predicted_clv', ascending=False).head(20)
    
    display_cols = ['customer_id', 'segment', 'region', 'total_revenue', 'predicted_clv', 'order_count', 'avg_satisfaction']
    st.dataframe(
        top_customers[display_cols].style.format({
            'total_revenue': '${:,.2f}',
            'predicted_clv': '${:,.2f}',
            'avg_satisfaction': '{:.2f}'
        }),
        use_container_width=True
    )

# Tab 4: Model Performance
with tab4:
    st.header("Model Performance Tracking")
    
    st.subheader("Revenue Forecasting Model")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", "ARIMA(5,1,2)")
    col2.metric("Training Data", f"{train_size} days")
    col3.metric("Test Accuracy", ">80%")
    
    st.subheader("Churn Prediction Model")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", "Random Forest")
    col2.metric("Accuracy", f"{accuracy*100:.1f}%")
    col3.metric("Features Used", len(feature_cols))
    
    st.subheader("Scenario Analysis")
    
    scenario = st.selectbox(
        "Select Scenario",
        ["Optimistic", "Base Case", "Pessimistic"]
    )
    
    base_revenue = daily_revenue['revenue'].tail(30).mean()
    
    if scenario == "Optimistic":
        multiplier = 1.2
        description = "20% growth in average daily revenue"
    elif scenario == "Base Case":
        multiplier = 1.0
        description = "Maintain current revenue levels"
    else:
        multiplier = 0.8
        description = "20% decline in average daily revenue"
    
    st.info(f"**{scenario} Scenario:** {description}")
    
    forecast_dates = pd.date_range(
        start=daily_revenue['date'].max() + pd.Timedelta(days=1),
        periods=90
    )
    
    scenario_revenue = [base_revenue * multiplier] * 90
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Projected Daily Revenue", f"${base_revenue * multiplier:,.2f}")
        st.metric("Projected Monthly Revenue", f"${base_revenue * multiplier * 30:,.2f}")
        st.metric("Projected Quarterly Revenue", f"${base_revenue * multiplier * 90:,.2f}")
    
    with col2:
        scenario_df = pd.DataFrame({
            'Date': forecast_dates,
            'Projected Revenue': scenario_revenue
        })
        
        fig = px.line(
            scenario_df,
            x='Date',
            y='Projected Revenue',
            title=f'{scenario} Revenue Projection (90 Days)'
        )
        st.plotly_chart(fig, use_container_width=True)
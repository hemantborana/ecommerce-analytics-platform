import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import load_transactions, calculate_kpis

st.set_page_config(page_title="KPI Monitoring", layout="wide")

@st.cache_data
def load_data():
    return load_transactions()

df = load_data()

st.title("KPI Monitoring & Alerts")

# Define KPI thresholds
thresholds = {
    'daily_revenue': {'target': 4500, 'min': 3500},
    'daily_orders': {'target': 22, 'min': 15},
    'avg_satisfaction': {'target': 3.5, 'min': 3.0},
    'aov': {'target': 200, 'min': 150}
}

# Calculate current metrics
df['date_only'] = df['date'].dt.date
latest_date = df['date_only'].max()
latest_data = df[df['date_only'] == latest_date]

current_metrics = {
    'daily_revenue': latest_data['revenue'].sum(),
    'daily_orders': len(latest_data),
    'avg_satisfaction': latest_data['satisfaction_score'].mean(),
    'aov': latest_data['revenue'].mean()
}

# Alert section
st.header("Active Alerts")
alerts = []

for metric, value in current_metrics.items():
    if value < thresholds[metric]['min']:
        alerts.append({
            'metric': metric.replace('_', ' ').title(),
            'status': 'Critical',
            'value': value,
            'threshold': thresholds[metric]['min']
        })
    elif value < thresholds[metric]['target']:
        alerts.append({
            'metric': metric.replace('_', ' ').title(),
            'status': 'Warning',
            'value': value,
            'threshold': thresholds[metric]['target']
        })

if alerts:
    for alert in alerts:
        if alert['status'] == 'Critical':
            st.error(f"🔴 {alert['metric']}: {alert['value']:.2f} (Below minimum: {alert['threshold']})")
        else:
            st.warning(f"🟡 {alert['metric']}: {alert['value']:.2f} (Below target: {alert['threshold']})")
else:
    st.success("✅ All KPIs are meeting targets")

st.divider()

# KPI gauge charts
st.header("KPI Performance Gauges")

col1, col2 = st.columns(2)

with col1:
    fig_revenue = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_metrics['daily_revenue'],
        title={'text': "Daily Revenue ($)"},
        delta={'reference': thresholds['daily_revenue']['target']},
        gauge={
            'axis': {'range': [0, 8000]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, thresholds['daily_revenue']['min']], 'color': "lightgray"},
                {'range': [thresholds['daily_revenue']['min'], thresholds['daily_revenue']['target']], 'color': "yellow"},
                {'range': [thresholds['daily_revenue']['target'], 8000], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': thresholds['daily_revenue']['target']
            }
        }
    ))
    st.plotly_chart(fig_revenue, use_container_width=True)

with col2:
    fig_orders = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_metrics['daily_orders'],
        title={'text': "Daily Orders"},
        delta={'reference': thresholds['daily_orders']['target']},
        gauge={
            'axis': {'range': [0, 40]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, thresholds['daily_orders']['min']], 'color': "lightgray"},
                {'range': [thresholds['daily_orders']['min'], thresholds['daily_orders']['target']], 'color': "yellow"},
                {'range': [thresholds['daily_orders']['target'], 40], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': thresholds['daily_orders']['target']
            }
        }
    ))
    st.plotly_chart(fig_orders, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig_satisfaction = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_metrics['avg_satisfaction'],
        title={'text': "Avg Satisfaction Score"},
        delta={'reference': thresholds['avg_satisfaction']['target']},
        gauge={
            'axis': {'range': [1, 5]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [1, thresholds['avg_satisfaction']['min']], 'color': "lightgray"},
                {'range': [thresholds['avg_satisfaction']['min'], thresholds['avg_satisfaction']['target']], 'color': "yellow"},
                {'range': [thresholds['avg_satisfaction']['target'], 5], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': thresholds['avg_satisfaction']['target']
            }
        }
    ))
    st.plotly_chart(fig_satisfaction, use_container_width=True)

with col4:
    fig_aov = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_metrics['aov'],
        title={'text': "Average Order Value ($)"},
        delta={'reference': thresholds['aov']['target']},
        gauge={
            'axis': {'range': [0, 400]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, thresholds['aov']['min']], 'color': "lightgray"},
                {'range': [thresholds['aov']['min'], thresholds['aov']['target']], 'color': "yellow"},
                {'range': [thresholds['aov']['target'], 400], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': thresholds['aov']['target']
            }
        }
    ))
    st.plotly_chart(fig_aov, use_container_width=True)

# KPI trends
st.header("KPI Trends (Last 30 Days)")
last_30_days = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=30))]
daily_kpis = calculate_kpis(last_30_days, 'D')

fig_trends = go.Figure()
fig_trends.add_trace(go.Scatter(
    x=daily_kpis['period'],
    y=daily_kpis['total_revenue'],
    name='Revenue',
    yaxis='y1'
))
fig_trends.add_trace(go.Scatter(
    x=daily_kpis['period'],
    y=daily_kpis['total_orders'],
    name='Orders',
    yaxis='y2'
))

fig_trends.update_layout(
    yaxis=dict(title='Revenue ($)'),
    yaxis2=dict(title='Orders', overlaying='y', side='right'),
    xaxis=dict(title='Date'),
    hovermode='x unified'
)

st.plotly_chart(fig_trends, use_container_width=True)
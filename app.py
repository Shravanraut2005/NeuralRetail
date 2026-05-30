# NeuralRetail | Amdox Internship
# Streamlit Dashboard
# Author: Shravan

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralRetail Dashboard",
    page_icon="📊",
    layout="wide"
)

# ─── LOAD DATA ───────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/online_retail_featured.csv")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    rfm = pd.read_csv("data/features/rfm_segments.csv")
    daily = pd.read_csv("data/features/daily_sales.csv")
    daily['Date'] = pd.to_datetime(daily['Date'])
    forecast = pd.read_csv("data/features/revenue_forecast.csv")
    forecast['Date'] = pd.to_datetime(forecast['Date'])
    inventory = pd.read_csv("data/features/inventory_plan.csv")
    return df, rfm, daily, forecast, inventory

df, rfm, daily, forecast, inventory = load_data()

# ─── SIDEBAR ─────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=80)
st.sidebar.title("NeuralRetail")
st.sidebar.markdown("AI Powered Retail Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Sales Overview",
    "👥 Customer Segmentation",
    "📈 Demand Forecasting",
    "📦 Inventory Optimization"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Stats**")
st.sidebar.metric("Total Transactions", f"{len(df):,}")
st.sidebar.metric("Total Customers", f"{rfm.shape[0]:,}")
st.sidebar.metric("Total Revenue", f"£{df['TotalRevenue'].sum():,.0f}")

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — SALES OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Sales Overview":
    st.title("📊 Sales Overview")
    st.markdown("Complete view of revenue, products and geographic performance.")
    st.markdown("---")

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue",    f"£{df['TotalRevenue'].sum():,.0f}")
    col2.metric("Total Orders",     f"{df['Invoice'].nunique():,}")
    col3.metric("Total Products",   f"{df['StockCode'].nunique():,}")
    col4.metric("Total Countries",  f"{df['Country'].nunique()}")
    st.markdown("---")

    # Monthly Revenue
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Revenue Trend")
        monthly = df.groupby(['Year','Month'])['TotalRevenue'].sum().reset_index()
        monthly['YearMonth'] = monthly['Year'].astype(str) + '-' + \
                               monthly['Month'].astype(str).str.zfill(2)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(monthly['YearMonth'], monthly['TotalRevenue'],
                marker='o', color='steelblue', linewidth=2)
        ax.fill_between(range(len(monthly)), monthly['TotalRevenue'],
                        alpha=0.15, color='steelblue')
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels(monthly['YearMonth'], rotation=45, ha='right', fontsize=7)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k"))
        ax.set_title("Monthly Revenue", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Revenue by Day of Week")
        day_names = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
        dow = df.groupby('DayOfWeek')['TotalRevenue'].sum().reset_index()
        dow['Day'] = dow['DayOfWeek'].map(day_names)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#e74c3c' if d == dow['TotalRevenue'].idxmax()
                  else 'steelblue' for d in range(len(dow))]
        ax.bar(dow['Day'], dow['TotalRevenue'], color=colors, edgecolor='white')
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k"))
        ax.set_title("Revenue by Day", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Top 10 Products by Revenue")
        top_prod = df.groupby('Description')['TotalRevenue'].sum()\
                     .sort_values(ascending=True).tail(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top_prod.index, top_prod.values,
                color='steelblue', edgecolor='white')
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k"))
        ax.set_title("Top 10 Products", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Top 10 Countries by Revenue")
        top_countries = df.groupby('Country')['TotalRevenue'].sum()\
                          .sort_values(ascending=True).tail(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top_countries.index, top_countries.values,
                color='#e67e22', edgecolor='white')
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k"))
        ax.set_title("Top 10 Countries", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER SEGMENTATION
# ═══════════════════════════════════════════════════════════════
elif page == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation")
    st.markdown("RFM-based customer segments using KMeans clustering.")
    st.markdown("---")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Champions",          f"{len(rfm[rfm['Segment']=='Champions']):,}")
    col2.metric("Potential Loyalists",f"{len(rfm[rfm['Segment']=='Potential Loyalists']):,}")
    col3.metric("At Risk",            f"{len(rfm[rfm['Segment']=='At Risk']):,}")
    col4.metric("Lost Customers",     f"{len(rfm[rfm['Segment']=='Lost Customers']):,}")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Segment Distribution")
        seg_counts = rfm['Segment'].value_counts()
        colors = ['#2ecc71','#3498db','#e67e22','#e74c3c']
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(seg_counts.values, labels=seg_counts.index,
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title("Customer Segments", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Average Spend per Segment")
        seg_mon = rfm.groupby('Segment')['Monetary'].mean()\
                     .sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.barh(seg_mon.index, seg_mon.values,
                color=['#e74c3c','#e67e22','#3498db','#2ecc71'],
                edgecolor='white')
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
        ax.set_title("Avg Spend per Segment", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Recency Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        for seg, color in zip(['Champions','At Risk','Lost Customers'],
                              ['#2ecc71','#e67e22','#e74c3c']):
            data = rfm[rfm['Segment']==seg]['Recency']
            ax.hist(data, bins=30, alpha=0.6, label=seg, color=color)
        ax.set_xlabel("Days since last purchase")
        ax.set_ylabel("Customers")
        ax.legend()
        ax.set_title("Recency by Segment", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("RFM Summary Table")
        summary = rfm.groupby('Segment')[['Recency','Frequency','Monetary']]\
                     .mean().round(2)
        st.dataframe(summary, use_container_width=True)
        st.markdown("---")
        st.subheader("Churn Rate by Segment")
        churn_seg = rfm.groupby('Segment')['Churned'].mean().round(3)*100
        st.dataframe(churn_seg.rename("Churn Rate (%)"),
                     use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — DEMAND FORECASTING
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Demand Forecasting":
    st.title("📈 Demand Forecasting")
    st.markdown("Prophet model — 90 day revenue forecast.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Historical Daily Rev", f"£{daily['Revenue'].mean():,.0f}")
    col2.metric("Avg Forecast Daily Rev",
                f"£{forecast['Predicted_Revenue'].mean():,.0f}")
    col3.metric("Max Forecast Daily Rev",
                f"£{forecast['Predicted_Revenue'].max():,.0f}")
    st.markdown("---")

    # Weekly aggregation
    hist_weekly = daily.resample('W', on='Date')['Revenue'].sum().reset_index()
    fc_weekly = forecast.resample('W', on='Date').agg({
        'Predicted_Revenue':'sum',
        'Lower_Bound':'sum',
        'Upper_Bound':'sum'
    }).reset_index()

    st.subheader("Weekly Revenue — Historical + 90 Day Forecast")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hist_weekly['Date'], hist_weekly['Revenue'],
            color='steelblue', linewidth=2, label='Actual Revenue')
    ax.fill_between(fc_weekly['Date'],
                    fc_weekly['Lower_Bound'],
                    fc_weekly['Upper_Bound'],
                    alpha=0.3, color='orange')
    ax.plot(fc_weekly['Date'], fc_weekly['Predicted_Revenue'],
            color='orange', linewidth=2.5,
            marker='o', markersize=5, label='Forecast')
    ax.axvline(x=daily['Date'].max(), color='red',
               linestyle='--', linewidth=1.5, label='Forecast Start')
    ax.set_ylabel("Revenue (£)")
    ax.legend()
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k"))
    ax.set_title("Revenue Forecast", fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Forecast Data Table")
        st.dataframe(forecast.head(20), use_container_width=True)

    with col2:
        st.subheader("Monthly Forecast Summary")
        forecast['Month'] = forecast['Date'].dt.to_period('M').astype(str)
        monthly_fc = forecast.groupby('Month')['Predicted_Revenue']\
                             .sum().reset_index()
        monthly_fc.columns = ['Month','Predicted Revenue (£)']
        monthly_fc['Predicted Revenue (£)'] = \
            monthly_fc['Predicted Revenue (£)'].round(2)
        st.dataframe(monthly_fc, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — INVENTORY OPTIMIZATION
# ═══════════════════════════════════════════════════════════════
elif page == "📦 Inventory Optimization":
    st.title("📦 Inventory Optimization")
    st.markdown("EOQ-based reorder points and safety stock for all products.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Products Analysed", f"{len(inventory):,}")
    col2.metric("Avg Reorder Point",
                f"{inventory['Reorder_Point'].mean():,.0f} units")
    col3.metric("Avg EOQ",
                f"{inventory['EOQ'].mean():,.0f} units")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 15 — Reorder Points")
        top15 = inventory.head(15).copy()
        top15['Name'] = top15['Description'].str[:22]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top15['Name'], top15['Reorder_Point'],
                color='steelblue', edgecolor='white')
        ax.set_xlabel("Units")
        ax.set_title("Reorder Points", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Top 15 — Economic Order Quantity")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top15['Name'], top15['EOQ'],
                color='#2ecc71', edgecolor='white')
        ax.set_xlabel("Units to Order")
        ax.set_title("EOQ per Product", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Full Inventory Plan — Search & Filter")
    search = st.text_input("Search product name:")
    if search:
        filtered = inventory[inventory['Description'].str.contains(
            search, case=False, na=False)]
    else:
        filtered = inventory.head(50)

    st.dataframe(filtered[[
        'Description','Daily_Demand','EOQ',
        'Safety_Stock','Reorder_Point','Avg_Price','Total_Revenue'
    ]].round(2), use_container_width=True)
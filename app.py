import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="E-Commerce Executive Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# COLOR THEME
# ==========================================================
PRIMARY = "#4F46E5"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
PURPLE = "#9333EA"

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Sales_data.csv", encoding="latin-1")

    if 'ï»¿Order No' in df.columns:
        df.rename(columns={'ï»¿Order No': 'Order No'}, inplace=True)

    df['Total'] = df['Total'].astype(str).str.replace('[$,]', '', regex=True)
    df['Total'] = pd.to_numeric(df['Total'], errors='coerce')

    df['Retail Price'] = df['Retail Price'].astype(str).str.replace('[$,]', '', regex=True)
    df['Retail Price'] = pd.to_numeric(df['Retail Price'], errors='coerce')

    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')

    df = df.dropna(subset=['Order Date', 'Total'])
    return df


df = load_data()

# ==========================================================
# SIDEBAR FILTERS
# ==========================================================
st.sidebar.header("🔎 Filters")

min_date = df['Order Date'].min()
max_date = df['Order Date'].max()

start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

selected_customers = st.sidebar.multiselect(
    "Select Customer(s)",
    options=sorted(df['Customer Name'].dropna().unique()),
    default=sorted(df['Customer Name'].dropna().unique())
)

selected_products = st.sidebar.multiselect(
    "Select Product(s)",
    options=sorted(df['Product Name'].dropna().unique()),
    default=sorted(df['Product Name'].dropna().unique())
)

filtered_df = df[
    (df['Order Date'] >= pd.to_datetime(start_date)) &
    (df['Order Date'] <= pd.to_datetime(end_date)) &
    (df['Customer Name'].isin(selected_customers)) &
    (df['Product Name'].isin(selected_products))
]

if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# ==========================================================
# RFM ANALYSIS
# ==========================================================
rfm_data = (
    filtered_df
    .groupby('Customer Name')
    .agg({
        'Order Date': lambda x: (filtered_df['Order Date'].max() - x.max()).days,
        'Order No': 'nunique',
        'Total': 'sum'
    })
    .reset_index()
)

rfm_data.columns = ['Customer', 'Recency', 'Frequency', 'Monetary']

# Handle small dataset safely
if len(rfm_data) >= 5:

    rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'], 5, labels=[5,4,3,2,1])
    rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'], 5, labels=[1,2,3,4,5])

    rfm_data['RFM_Score'] = (
        rfm_data['R_Score'].astype(int) +
        rfm_data['F_Score'].astype(int) +
        rfm_data['M_Score'].astype(int)
    )
else:
    rfm_data['RFM_Score'] = 0

avg_rfm = rfm_data['RFM_Score'].mean()

# ==========================================================
# HEADER
# ==========================================================
st.title("📊 E-Commerce Executive Dashboard")
st.markdown("### Dynamic Sales, Customer Segmentation & RFM Insights")
st.divider()

# ==========================================================
# KPI SECTION
# ==========================================================
st.markdown("## 📌 Key Performance Indicators")

total_revenue = filtered_df['Total'].sum()
total_orders = filtered_df['Order No'].nunique()
avg_order_value = filtered_df['Total'].mean()
active_customers = filtered_df['Customer Name'].nunique()

revenue_color = SUCCESS if total_revenue > 50000 else WARNING

col1, col2, col3, col4, col5 = st.columns(5)

def kpi_card(title, value, color):
    return f"""
        <div style="
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            text-align: center;
        ">
            <h4 style="color:#111111; margin-bottom:10px;">
                {title}
            </h4>
            <h2 style="color:{color}; margin:0;">
                {value}
            </h2>
        </div>
    """

with col1:
    st.markdown(kpi_card("Total Revenue", f"${total_revenue:,.0f}", revenue_color), unsafe_allow_html=True)

with col2:
    st.markdown(kpi_card("Total Orders", total_orders, PRIMARY), unsafe_allow_html=True)

with col3:
    st.markdown(kpi_card("Avg Order Value", f"${avg_order_value:,.2f}", PRIMARY), unsafe_allow_html=True)

with col4:
    st.markdown(kpi_card("Active Customers", active_customers, PRIMARY), unsafe_allow_html=True)

with col5:
    st.markdown(kpi_card("Avg RFM Score", f"{avg_rfm:.2f}", PURPLE), unsafe_allow_html=True)

st.divider()

# ==========================================================
# CHARTS
# ==========================================================
colA, colB = st.columns(2)

# Revenue Trend
with colA:
    st.subheader("📈 Revenue Trend")

    revenue_trend = (
        filtered_df
        .groupby('Order Date')['Total']
        .sum()
        .reset_index()
    )

    fig1 = px.area(
        revenue_trend,
        x='Order Date',
        y='Total',
        template="plotly_white",
        color_discrete_sequence=[PRIMARY]
    )

    st.plotly_chart(fig1, use_container_width=True)

# Top Products
with colB:
    st.subheader("🏆 Top 10 Products")

    top_products = (
        filtered_df
        .groupby('Product Name')['Total']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig2 = px.bar(
        top_products,
        x='Total',
        y='Product Name',
        orientation='h',
        color='Total',
        color_continuous_scale='Blues',
        template="plotly_white"
    )

    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ==========================================================
# CUSTOMER SEGMENTATION (KMEANS)
# ==========================================================
st.subheader("👥 Customer Segmentation")

customer_data = (
    filtered_df
    .groupby('Customer Name')
    .agg({'Total': 'sum', 'Order No': 'nunique'})
    .reset_index()
)

customer_data.columns = ['Customer', 'TotalSpent', 'Orders']

if len(customer_data) >= 3:

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(customer_data[['TotalSpent', 'Orders']])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customer_data['Cluster'] = kmeans.fit_predict(scaled_data)

    fig3 = px.scatter(
        customer_data,
        x='TotalSpent',
        y='Orders',
        color=customer_data['Cluster'].astype(str),
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data=['Customer'],
        template="plotly_white"
    )

    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ==========================================================
# RFM SEGMENTATION CHART
# ==========================================================
if len(rfm_data) >= 5:

    st.subheader("🎯 RFM Customer Segments")

    rfm_data['Segment'] = pd.cut(
        rfm_data['RFM_Score'],
        bins=[0,6,9,12,15],
        labels=["Low Value","Mid Value","High Value","Top Customers"]
    )

    segment_count = rfm_data['Segment'].value_counts().reset_index()
    segment_count.columns = ['Segment', 'Count']

    fig_rfm = px.bar(
        segment_count,
        x='Segment',
        y='Count',
        color='Segment',
        template="plotly_white"
    )

    st.plotly_chart(fig_rfm, use_container_width=True)

# ==========================================================
# DATA TABLE
# ==========================================================
with st.expander("📄 View Filtered Data"):
    st.dataframe(filtered_df, use_container_width=True)

# ==========================================================
# DOWNLOAD BUTTON
# ==========================================================
st.sidebar.download_button(
    "📥 Download Filtered Data",
    filtered_df.to_csv(index=False),
    "filtered_sales_data.csv",
    "text/csv"
)

# WEB

print(rf.feature_names_in_)


!pip install streamlit pyngrok --quiet


from pyngrok import ngrok
ngrok.set_auth_token("34EefkCGuMf5ZpBRYM1E632fN2B_3gBuM2qpiPE18cMMPsaj6")


%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from datetime import datetime


st.set_page_config(page_title="RETAIL SALES FORECASTING FOR ROSSMAN STORE ", page_icon="💠", layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =======================
# LOAD MODEL
# =======================
try:
    model = joblib.load("lightgbm_sales_model.pkl")
except:
    st.warning("⚠️ Không tìm thấy mô hình `lightgbm_sales_model.pkl`. Vui lòng tải lên trước khi chạy app.")

# =======================
# HEADER
# =======================
st.markdown("""
<div class="glass-header">
    <h1>💠 ROSSMAN STORE</h1>
    <p>Dự đoán doanh số 6 và 12 tháng tới bằng mô hình LightGBM</p>
</div>
""", unsafe_allow_html=True)

# =======================
# FORM INPUT
# =======================
with st.sidebar:
    st.markdown("<h2 class='sidebar-title'>📋 Nhập thông tin cửa hàng</h2>", unsafe_allow_html=True)
    store_id = st.number_input("🏬 Mã cửa hàng", min_value=1, step=1)
    current_sales = st.number_input("💰 Doanh số hiện tại (USD)", min_value=0.0, step=100.0)
    competition_distance = st.number_input("📏 Khoảng cách đối thủ (m)", min_value=0.0, step=10.0)
    promo = st.selectbox("🎟️ Đang khuyến mãi?", ["Không", "Có"])
    customers = st.number_input("👥 Số khách trung bình/ngày", min_value=0, step=1)
    is_holiday = st.selectbox("🎄 Mùa lễ hội?", ["Không", "Có"])
    is_weekend = st.selectbox("🗓️ Cuối tuần?", ["Không", "Có"])
    run_button = st.button("🚀 DỰ ĐOÁN DOANH SỐ")

# =======================
# TẠO DỮ LIỆU ĐẦU VÀO
# =======================
input_data = {
    'Store': store_id,
    'Promo': 1 if promo == "Có" else 0,
    'CompetitionDistance': competition_distance,
    'CompetitionActive': 1,
    'CompetitionMonths': 12,
    'Sales_Lag1': current_sales * 0.9,
    'Sales_Lag2': current_sales * 0.95,
    'Sales_Lag3': current_sales,
    'Month': datetime.now().month,
    'Quarter': (datetime.now().month - 1)//3 + 1,
    'Year': datetime.now().year,
    'IsWeekend': 1 if is_weekend == "Có" else 0,
    'IsHolidaySeason': 1 if is_holiday == "Có" else 0,
    'Customers': customers,
    'DaysAhead': 0
}
df_input = pd.DataFrame([input_data])

# =======================
# DỰ ĐOÁN
# =======================
if run_button:
    df_6m = df_input.copy()
    df_12m = df_input.copy()
    df_6m['DaysAhead'] = 180
    df_12m['DaysAhead'] = 365

    pred_6m = model.predict(df_6m)[0]
    pred_12m = model.predict(df_12m)[0]

    growth_6m = ((pred_6m - current_sales) / current_sales * 100) if current_sales > 0 else 0
    growth_12m = ((pred_12m - current_sales) / current_sales * 100) if current_sales > 0 else 0

    st.markdown("<h2 class='section-title'>📈 KẾT QUẢ DỰ ĐOÁN</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='glass-card'><p>Hiện tại</p><h3>{current_sales:,.0f} USD</h3></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='glass-card green'><p>6 tháng tới</p><h3>{pred_6m:,.0f} USD</h3><span>+{growth_6m:.2f}%</span></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='glass-card blue'><p>12 tháng tới</p><h3>{pred_12m:,.0f} USD</h3><span>+{growth_12m:.2f}%</span></div>", unsafe_allow_html=True)

    # =======================
    # BIỂU ĐỒ GLASS
    # =======================
    chart_data = pd.DataFrame({
        'Thời điểm': ['Hiện tại', '6 tháng tới', '12 tháng tới'],
        'Doanh số': [current_sales, pred_6m, pred_12m]
    })

    fig = px.bar(chart_data, x='Thời điểm', y='Doanh số',
                 color='Thời điểm',
                 text='Doanh số',
                 color_discrete_sequence=['#E57373', '#81C784', '#64B5F6'])
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig.update_layout(
        title="📊 So sánh doanh số dự đoán",
        template="plotly_white",
        height=450,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Nhập thông tin ở thanh bên trái và nhấn **DỰ ĐOÁN DOANH SỐ** để xem kết quả.")

!kill -9 $(lsof -t -i:8501)


from pyngrok import ngrok
public_url = ngrok.connect(8501)
print("🌐 Mở web tại:", public_url)
!streamlit run app.py --server.port 8501


print(rf.feature_names_in_)


!pip install streamlit pyngrok --quiet


from pyngrok import ngrok
ngrok.set_auth_token("34EefkCGuMf5ZpBRYM1E632fN2B_3gBuM2qpiPE18cMMPsaj6")


%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

lgbm = joblib.load("lightgbm_sales_model.pkl")

st.set_page_config(page_title="Dự báo doanh số bán lẻ", page_icon="📈", layout="centered")
st.title("📊 DỰ BÁO DOANH SỐ BÁN LẺ (LightGBM)")
st.markdown("Dự đoán doanh số **6 tháng** và **12 tháng tới** dựa trên thông tin cửa hàng và các yếu tố kinh doanh.")

store_id = st.number_input("Mã cửa hàng", min_value=1, step=1)
current_sales = st.number_input("Doanh số hiện tại (trung bình)", min_value=0.0, step=100.0)
competition_distance = st.number_input("Khoảng cách đối thủ (m)", min_value=0.0, step=10.0)
promo = st.selectbox("Đang khuyến mãi?", ["Không", "Có"])
customers = st.number_input("Số lượng khách trung bình", min_value=0, step=1)
is_holiday = st.selectbox("Có trong mùa lễ hội?", ["Không", "Có"])
is_weekend = st.selectbox("Cuối tuần?", ["Không", "Có"])

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

if customers < 5:
    st.warning("⚠️ Số lượng khách trung bình quá thấp, kết quả dự đoán có thể không chính xác.")

features = list(input_data.keys())
df_input = pd.DataFrame([input_data])

if st.button("🔮 Dự đoán doanh số"):
    df_6m = df_input.copy()
    df_6m['DaysAhead'] = 180
    df_12m = df_input.copy()
    df_12m['DaysAhead'] = 365

    pred_6m = lgbm.predict(df_6m)[0]
    pred_12m = lgbm.predict(df_12m)[0]

    growth_6m = ((pred_6m - current_sales) / current_sales * 100) if current_sales > 0 else 0
    growth_12m = ((pred_12m - current_sales) / current_sales * 100) if current_sales > 0 else 0

    st.subheader("📈 KẾT QUẢ DỰ ĐOÁN")
    st.write(f"**Doanh số 6 tháng tới:** {pred_6m:,.2f}")
    st.write(f"**Doanh số 12 tháng tới:** {pred_12m:,.2f}")
    st.write(f"**Tăng trưởng 6 tháng:** {growth_6m:.2f}%")
    st.write(f"**Tăng trưởng 12 tháng:** {growth_12m:.2f}%")

    import matplotlib.pyplot as plt

# Tạo DataFrame biểu đồ
chart_data = pd.DataFrame({
    'Thời điểm': ['Hiện tại', '6 tháng tới', '12 tháng tới'],
    'Doanh số': [current_sales, pred_6m, pred_12m]
})

# Vẽ biểu đồ cột dọc
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(chart_data['Thời điểm'], chart_data['Doanh số'],
              color=['#ff4d4d', '#66b3ff', '#0066cc'])

# Thêm giá trị lên đầu cột
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 50, f"{height:,.0f}",
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel("Doanh số (VNĐ)")
ax.set_title("📊 So sánh doanh số hiện tại - 6 tháng - 12 tháng tới")
st.pyplot(fig)



!kill -9 $(lsof -t -i:8501)


from pyngrok import ngrok
public_url = ngrok.connect(8501)
print("🌐 Mở web tại:", public_url)
!streamlit run app.py --server.port 8501


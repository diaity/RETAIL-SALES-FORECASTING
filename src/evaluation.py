## SO SÁNH 3 MÔ HÌNH

# RMSE (càng nhỏ càng tốt)
best_model_name = None
best_model = None
best_rmse = float('inf')

models = {
    'Random Forest': (rf, rmse),
    'LightGBM': (lgbm, rmse_lgbm),
    'Linear Regression': (lr, rmse_lr)
}

for name, (model, rmse) in models.items():
    if rmse < best_rmse:
        best_rmse = rmse
        best_model_name = name
        best_model = model

print(f"\nMô hình tốt nhất: {best_model_name} (RMSE = {best_rmse:.4f})")


Mô hình LightGBM cho kết quả tốt hơn với RMSE 546.9191

Mô hình Random Forest và Linear Regression có sai số lớn hơn và khả năng giải thích dữ liệu thấp hơn so với LightGBM

Mặc dù mô hình LightGBM có phần giống Random Forest nhưng mô hình LightGBM tối ưu hơn, nhanh hơn, chính xác hơn, còn Linear Regression chỉ phù hợp để xem xu hướng tổng thể

# DỰ ĐOÁN DOANH SỐ

## 6 THÁNG KẾ TIẾP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

last_date = pd.to_datetime(df['Date'].max())
future_6m = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=180, freq='D')

def tao_dac_trung_tuong_lai(future_dates):
    df_future = pd.DataFrame({'Date': future_dates})
    df_future['Year'] = df_future['Date'].dt.year
    df_future['Month'] = df_future['Date'].dt.month
    df_future['Quarter'] = df_future['Date'].dt.quarter
    df_future['IsWeekend'] = df_future['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    df_future['IsHolidaySeason'] = df_future['Month'].isin([11, 12]).astype(int)
    df_future['DaysAhead'] = (df_future['Date'] - last_date).dt.days
    return df_future

future_6m_full = tao_dac_trung_tuong_lai(future_6m)

future_6m_full['DaysAhead'] = (future_6m_full['Date'] - df['Date'].max()).dt.days


for col in features:
    if col not in future_6m_full.columns:
        if col in df.columns:
            val = df[col].median() if np.issubdtype(df[col].dtype, np.number) else df[col].mode()[0]
            future_6m_full[col] = val
        else:
            future_6m_full[col] = 0

future_6m_full = future_6m_full[features]


# if customers < 5:
#     customers = 5

ket_qua = []
for store_id in df['Store'].unique():
    df_store = future_6m_full.copy()
    df_store['Store'] = store_id

    du_doan = lgbm.predict(df_store).mean()
    du_doan_tb = pd.Series(du_doan).groupby(df_store['Month']).mean().mean()

    hien_tai = df.loc[df['Store'] == store_id, 'Sales'].mean()

    ty_le_tang = ((du_doan - hien_tai) / hien_tai) * 100

    ket_qua.append([store_id, hien_tai, du_doan, ty_le_tang])

du_bao_6_thang = pd.DataFrame(ket_qua, columns=[
    'Cửa hàng', 'Doanh số hiện tại', 'Doanh số dự đoán 6 tháng tới', 'Tỷ lệ tăng trưởng (%)'
])
du_bao_6_thang[['Doanh số hiện tại', 'Doanh số dự đoán 6 tháng tới']] = du_bao_6_thang[
    ['Doanh số hiện tại', 'Doanh số dự đoán 6 tháng tới']
].round(2)
du_bao_6_thang['Tỷ lệ tăng trưởng (%)'] = du_bao_6_thang['Tỷ lệ tăng trưởng (%)'].round(2)

du_bao_6_thang_mau = du_bao_6_thang.sort_values("Cửa hàng").head(10)

print("DỰ BÁO DOANH SỐ 6 THÁNG TỚI\n")
print(du_bao_6_thang_mau.to_string(index=False))

plt.figure(figsize=(10,5))
bar_width = 0.35
x = np.arange(len(du_bao_6_thang_mau))

plt.bar(x - bar_width/2, du_bao_6_thang_mau['Doanh số hiện tại'],
        width=bar_width, label='Hiện tại', color='steelblue')
plt.bar(x + bar_width/2, du_bao_6_thang_mau['Doanh số dự đoán 6 tháng tới'],
        width=bar_width, label='Dự đoán', color='orange')

plt.xticks(x, du_bao_6_thang_mau['Cửa hàng'], rotation=0)
plt.title("So sánh doanh số hiện tại và dự đoán 6 tháng tới")
plt.xlabel("Cửa hàng")
plt.ylabel("Doanh số (trung bình)")
plt.legend()
plt.tight_layout()
plt.show()

Biểu đồ cho thấy doanh số trong 6 tháng tới có xu hướng tăng nhẹ và giữ ổn định so với doanh số hiện tại của các cửa hàng

Một số cửa hàng tăng rõ rệt, trong khi các cửa hàng khác giữ mức doanh thu tương đối ổn định, cho thấy mô hình dự đoán khả quan, ít biến động trong giai đoạn sắp tới

## 12 THÁNG KẾ TIẾP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

last_date = pd.to_datetime(df['Date'].max())
future_12m = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365, freq='D')

def tao_dac_trung_tuong_lai(future_dates):
    df_future = pd.DataFrame({'Date': future_dates})
    df_future['Year'] = df_future['Date'].dt.year
    df_future['Month'] = df_future['Date'].dt.month
    df_future['Quarter'] = df_future['Date'].dt.quarter
    df_future['IsWeekend'] = df_future['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    df_future['IsHolidaySeason'] = df_future['Month'].isin([11, 12]).astype(int)
    df_future['DaysAhead'] = (df_future['Date'] - last_date).dt.days

    return df_future

future_12m_full = tao_dac_trung_tuong_lai(future_12m)

for col in features:
    if col not in future_12m_full.columns:
        if col in df.columns:
            val = df[col].median() if np.issubdtype(df[col].dtype, np.number) else df[col].mode()[0]
            future_12m_full[col] = val
        else:
            future_12m_full[col] = 0

future_12m_full = future_12m_full[features]

# if customers < 5:
#     customers = 5


ket_qua = []
for store_id in df['Store'].unique():
    df_store = future_12m_full.copy()
    df_store['Store'] = store_id

    du_doan = lgbm.predict(df_store).mean()
    du_doan_tb = pd.Series(du_doan).groupby(df_store['Month']).mean().mean()

    hien_tai = df.loc[df['Store'] == store_id, 'Sales'].mean()
    ty_le_tang = ((du_doan - hien_tai) / hien_tai) * 100
    ket_qua.append([store_id, hien_tai, du_doan, ty_le_tang])

du_bao_12_thang = pd.DataFrame(ket_qua, columns=[
    'Cửa hàng', 'Doanh số hiện tại', 'Doanh số dự đoán 12 tháng tới', 'Tỷ lệ tăng trưởng (%)'
])
du_bao_12_thang[['Doanh số hiện tại', 'Doanh số dự đoán 12 tháng tới']] = du_bao_12_thang[
    ['Doanh số hiện tại', 'Doanh số dự đoán 12 tháng tới']
].round(2)
du_bao_12_thang['Tỷ lệ tăng trưởng (%)'] = du_bao_12_thang['Tỷ lệ tăng trưởng (%)'].round(2)

du_bao_12_thang_mau = du_bao_12_thang.sort_values("Cửa hàng").head(10)

print("DỰ BÁO DOANH SỐ 12 THÁNG TỚI\n")
print(du_bao_12_thang_mau.to_string(index=False))

plt.figure(figsize=(10,5))
bar_width = 0.35
x = np.arange(len(du_bao_12_thang_mau))

plt.bar(x - bar_width/2, du_bao_12_thang_mau['Doanh số hiện tại'],
        width=bar_width, label='Hiện tại', color='steelblue')
plt.bar(x + bar_width/2, du_bao_12_thang_mau['Doanh số dự đoán 12 tháng tới'],
        width=bar_width, label='Dự đoán', color='orange')

plt.xticks(x, du_bao_12_thang_mau['Cửa hàng'], rotation=0)
plt.title("So sánh doanh số hiện tại và dự đoán 12 tháng tới")
plt.xlabel("Cửa hàng")
plt.ylabel("Doanh số (trung bình)")
plt.legend()
plt.tight_layout()
plt.show()

## NHẬN XÉT

import matplotlib.pyplot as plt
import numpy as np

merged = du_bao_6_thang.merge(
    du_bao_12_thang,
    on='Cửa hàng',
    suffixes=('_6m', '_12m')
)

sample = merged.sort_values("Cửa hàng").head(10)

x = np.arange(len(sample))
width = 0.35
plt.figure(figsize=(10,5))
plt.bar(x - width/2, sample['Doanh số dự đoán 6 tháng tới'], width, label='Dự đoán 6 tháng', color='#3498db')
plt.bar(x + width/2, sample['Doanh số dự đoán 12 tháng tới'], width, label='Dự đoán 12 tháng', color='#2ecc71')

plt.xticks(x, sample['Cửa hàng'])
plt.xlabel('Cửa hàng')
plt.ylabel('Doanh số dự đoán')
plt.title('So sánh doanh số dự đoán 6 và 12 tháng tới', fontsize=13, weight='bold')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


top10 = du_bao_6_thang.sort_values(by='Tỷ lệ tăng trưởng (%)', ascending=False).head(10)
plt.figure(figsize=(10,5))
plt.barh(top10['Cửa hàng'].astype(str), top10['Tỷ lệ tăng trưởng (%)'], color='limegreen')
plt.title("Top 10 cửa hàng tăng trưởng doanh số cao nhất (6 tháng tới)")
plt.xlabel("Tỷ lệ tăng trưởng (%)")
plt.ylabel("Cửa hàng")
plt.gca().invert_yaxis()
plt.show()


bottom10 = du_bao_6_thang.sort_values(by='Tỷ lệ tăng trưởng (%)').head(10)
plt.figure(figsize=(10,5))
plt.barh(bottom10['Cửa hàng'].astype(str), bottom10['Tỷ lệ tăng trưởng (%)'], color='red')
plt.title("Top 10 cửa hàng giảm doanh số mạnh nhất (6 tháng tới)")
plt.xlabel("Tỷ lệ tăng trưởng (%)")
plt.ylabel("Cửa hàng")
plt.gca().invert_yaxis()
plt.show()


Dựa vào các chỉ số tăng trưởng qua 6 và 12 tháng thì có thể rút ra các nhận xét như sau:

Các cửa hàng có chỉ số tăng trưởng dương: cần tăng lượng hàng tồn kho và các khuyến mãi

Các cửa hàng có chỉ số tăng trưởng âm: cần giảm lượng tồn kho, tránh tồn đọng

Nếu 6 tháng giảm mà 12 tháng tăng thì có thể là do dịp lễ, chu kỳ sản phẩm

Nếu trong 2 kỳ tăng thì cửa hàng ổn định lâu dài

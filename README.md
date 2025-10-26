# 🏪 RETAIL SALES FORECASTING – DỰ BÁO DOANH SỐ BÁN LẺ

---

## 📖 1. Giới thiệu
Dự án **Retail Sales Forecasting** nhằm dự đoán doanh số bán lẻ của các cửa hàng trong **6 tháng** và **12 tháng tới**,  
giúp hỗ trợ **quản lý hàng tồn kho** và **ra quyết định kinh doanh chính xác hơn**.

Bộ dữ liệu được sử dụng là **Rossmann Store Sales** từ Kaggle, bao gồm thông tin:
- Doanh số bán hàng theo ngày (`Sales`)
- Lượng khách hàng (`Customers`)
- Tình trạng khuyến mãi (`Promo`)
- Khoảng cách đến đối thủ (`CompetitionDistance`)
- Và các yếu tố thời gian (`Year`, `Month`, `Quarter`, `IsHolidaySeason`, `IsWeekend`)

Dự án được triển khai với 3 mô hình dự báo:
- **Random Forest Regressor**
- **LightGBM Regressor**
- **Linear Regression**

Cuối cùng, nhóm chọn **LightGBM** làm mô hình chính để tích hợp vào ứng dụng web Streamlit.

---

## 🧭 2. Mục tiêu
- Phân tích và xử lý dữ liệu bán lẻ thực tế.  
- Xây dựng các đặc trưng thời gian (feature engineering).  
- So sánh hiệu năng giữa ba mô hình dự báo.  
- Dự đoán doanh số trong 6 và 12 tháng tới.  
- Triển khai ứng dụng web dự báo doanh số trực quan.  

---

## ⚙️ 3. Cài đặt và chạy dự án
```bash
pip install -r requirements.txt
python src/model_training.py
cd app
streamlit run app.py
```

---

## 📊 4. Kết quả mô hình
| Mô hình | RMSE | R² | Nhận xét |
|----------|------|----|----------|
| Random Forest | 643.15 | 0.9696 | Ổn định, học phi tuyến |
| LightGBM | **546.92** | **0.9754** | Chính xác nhất, học nhanh |
| Linear Regression | 1244.35 | 0.84 | Đơn giản, tổng quan |

👉 Mô hình **LightGBM** được chọn để triển khai chính thức vì có **sai số thấp nhất** và **hiệu suất cao nhất**.

---

## 📈 5. Ứng dụng Web
- Nhập mã cửa hàng, doanh số, khuyến mãi, khách hàng,...  
- Dự đoán doanh số **6 tháng và 12 tháng tới**  
- Biểu đồ cột so sánh doanh thu thực tế & dự đoán.  

---

## 📚 6. Kết luận
Qua quá trình huấn luyện và so sánh, mô hình **LightGBM** cho kết quả tốt nhất  
với **RMSE = 546.92** và **R² = 0.9754**, vượt trội so với Random Forest và Linear Regression.  

👉 Điều này cho thấy LightGBM có khả năng:
- Xử lý dữ liệu lớn nhanh hơn.  
- Nắm bắt tốt các mối quan hệ phi tuyến giữa các biến.  
- Cho độ chính xác cao và ổn định hơn trong dự báo doanh số.

Ứng dụng dự báo doanh số giúp doanh nghiệp:
- Chủ động hơn trong việc **quản lý hàng tồn kho**.  
- **Lên kế hoạch khuyến mãi và nhân sự** hợp lý.  
- **Giảm thiểu rủi ro và tối ưu doanh thu**.

---

## 👥 7. Thành viên
| Họ tên | Vai trò |
|--------|----------|
| Nguyễn Tấn Đạt | Import dữ liệu, EDA, huấn luyện LightGBM, Linear, dự đoán doanh thu 6 tháng, web Streamlit |
| Võ Xuân Ân | Xử lý dữ liệu, huấn luyện mô hình Random Forest, dự đoán doanh thu 12 tháng, viết báo cáo |

---

## 📧 8. Liên hệ
> Nguyễn Tấn Đạt, Võ Xuân Ân  
> GitHub: [github.com/yourusername](https://github.com/diaity)  
> Email: nguyentandat19504@gmail.com

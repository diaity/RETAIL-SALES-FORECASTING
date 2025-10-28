# EDA

import matplotlib.pyplot as plt
import seaborn as sns

# Tổng quan
print("Tổng số cửa hàng:", train['Store'].nunique())
print("Doanh thu trung bình:", round(train['Sales'].mean(), 2))
print("Doanh thu cao nhất:", train['Sales'].max())
print("Doanh thu thấp nhất:", train['Sales'].min())


## PHÂN PHỐI DOANH THU

plt.figure(figsize=(8,5))
sns.histplot(train['Sales'], bins=50, kde=True)
plt.title("Phân phối doanh thu (Sales)")
plt.xlabel("Sales")
plt.ylabel("Tần suất")
plt.show()


## DOANH SỐ TRUNG BÌNH THEO THÁNG

plt.figure(figsize=(10,5))
sns.barplot(x='Month', y='Sales', data=train, estimator='mean', ci=None)
plt.title("Doanh số trung bình theo tháng")
plt.xlabel("Tháng")
plt.ylabel("Doanh số trung bình")
plt.show()


Cho thấy xu hướng doanh thu trong năm — thường tăng vào tháng 11–12 (cuối năm) và giảm nhẹ giữa năm.

## DOANH THU TRUNG BÌNH THEO NGÀY TRONG TUẦN

plt.figure(figsize=(8,5))
sns.barplot(x='DayOfWeek', y='Sales', data=train, estimator='mean', ci=None)
plt.title("Doanh số trung bình theo ngày trong tuần")
plt.xlabel("Thứ trong tuần (1=Thứ Hai, 7=Chủ nhật)")
plt.ylabel("Doanh số trung bình")
plt.show()


## DOANH THU TRUNG BÌNH THEO LOẠI CỬA HÀNG

plt.figure(figsize=(7,5))
sns.barplot(x='StoreType', y='Sales', data=train, estimator='mean', ci=None)
plt.title("Doanh số trung bình theo loại cửa hàng (StoreType)")
plt.xlabel("Loại cửa hàng")
plt.ylabel("Doanh số trung bình")
plt.show()


Các loại cửa hàng khác nhau (a, b, c, d) có quy mô và doanh số khác nhau — ví dụ, StoreType=c thường là siêu thị lớn nên có doanh số cao nhất.

## DOANH THU TRUNG BÌNH THEO NGÀY LỄ

plt.figure(figsize=(7,5))
sns.barplot(x='StateHoliday', y='Sales', data=train, estimator='mean', ci=None)
plt.title("Doanh số trung bình theo ngày lễ (StateHoliday)")
plt.xlabel("Loại ngày lễ (0=a,b,c)")
plt.ylabel("Doanh số trung bình")
plt.show()


## DOANH THU TRUNG BÌNH THEO MỨC ĐA DẠNG SẢN PHẨM

plt.figure(figsize=(7,5))
sns.barplot(x='Assortment', y='Sales', data=train, estimator='mean', ci=None)
plt.title("Doanh số trung bình theo mức đa dạng sản phẩm (Assortment)")
plt.xlabel("Assortment (a,b,c)")
plt.ylabel("Doanh số trung bình")
plt.show()


## KHOẢNG CÁCH VỚI CÁC ĐỐI THỦ ẢNH HƯỞNG ĐẾN DOANH THU

plt.figure(figsize=(8,5))
sns.scatterplot(x='CompetitionDistance', y='Sales', data=train, alpha=0.4)
plt.title("Quan hệ giữa khoảng cách cạnh tranh và doanh thu")
plt.xlabel("Khoảng cách tới đối thủ (mét)")
plt.ylabel("Doanh thu")
plt.xlim(0, 20000)
plt.show()


Cửa hàng có đối thủ càng gần thì doanh thu thường thấp hơn, thể hiện rõ trong phân tán dữ liệu.

## ẢNH HƯỞNG CỦA KHUYẾN MÃI ĐỐI VỚI DOANH THU

plt.figure(figsize=(7,5))
sns.barplot(x='Promo', y='Sales', data=train, estimator='mean', ci=None)
plt.title("Ảnh hưởng của chương trình khuyến mãi (Promo) đến doanh thu")
plt.xlabel("Có khuyến mãi hay không (0=Không, 1=Có)")
plt.ylabel("Doanh số trung bình")
plt.show()


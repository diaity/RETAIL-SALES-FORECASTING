# IMPORT DỮ LIỆU TẢI LÊN TỪ DRIVE

import pandas as pd

file_ids = {
    "train": "1VCFclTsvlmcbiOp2fNGT16u0Xn6Qz6_D",
    "test": "1DAVjQbd0m_bD0Cg34SnCyk2MbsqmjHJH",
    "store": "1PW4vAYlrMstMrI6DZGyyFVp9hAB-y0qf",
    "sample_submission": "16paq6-xFxisF5qfINcCzRfhHzuY50J7y"
}

train = pd.read_csv(f"https://drive.google.com/uc?id={file_ids['train']}")
test = pd.read_csv(f"https://drive.google.com/uc?id={file_ids['test']}")
store = pd.read_csv(f"https://drive.google.com/uc?id={file_ids['store']}")
sample_submission = pd.read_csv(f"https://drive.google.com/uc?id={file_ids['sample_submission']}")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Store shape:", store.shape)
print("Sample Submission shape:", sample_submission.shape)


## CẤU TRÚC CÁC CỘT TRONG BỘ DỮ LIỆU

print(train.head())


print(test.head())


print(store.head())


print(sample_submission.head())


# XỬ LÝ DỮ LIỆU

## CHUẨN HÓA DỮ LIỆU

# Một số cột như Store, Promo, Open, StateHoliday bị sai kiểu (object thay vì int)
# Chuẩn hóa kiểu dữ liệu
def normalize_types(df, is_train=True):
    # Cột số nguyên
    int_cols = ["Store", "DayOfWeek", "Open", "Promo", "SchoolHoliday", "Promo2", "Promo2SinceWeek", "Promo2SinceYear"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Chuyển StateHoliday về dạng chuỗi thống nhất
    if "StateHoliday" in df.columns:
        df["StateHoliday"] = df["StateHoliday"].astype(str).replace({"0": "0", "a": "a", "b": "b", "c": "c"})

    # Sales là số thực
    if is_train and "Sales" in df.columns:
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").astype("float32")

    return df

# Áp dụng cho tất cả các file
train = normalize_types(train, is_train=True)
test  = normalize_types(test, is_train=False)
store = normalize_types(store, is_train=False)


## GỘP DỮ LIỆU

# Xóa cột trùng trước khi gộp
dupe_cols = [c for c in store.columns if c in train.columns and c != 'Store']
train = train.drop(columns=dupe_cols, errors='ignore')
test = test.drop(columns=dupe_cols, errors='ignore')

# Gộp dữ liệu
train = pd.merge(train, store, on='Store', how='left')
test  = pd.merge(test, store, on='Store', how='left')

print("Train shape:", train.shape)
print("Test shape:", test.shape)


## XỬ LÝ GIÁ TRỊ THIẾU

# Xử lý giá trị thiếu trong train và test
for df in [train, test]:
    # Khoảng cách cạnh tranh (CompetitionDistance) Thay bằng median
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)

    # Năm & tuần bắt đầu khuyến mãi mở rộng (Promo2SinceYear / Promo2SinceWeek) Điền 0
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['Promo2SinceWeek'].fillna(0, inplace=True)

    # Các cửa hàng chưa có khuyến mãi thì để 'None'
    df['PromoInterval'].fillna('None', inplace=True)

    # Giá trị ngày lễ rỗng => không phải ngày lễ ('0')
    df['StateHoliday'].fillna('0', inplace=True)

    # Nếu cột Open bị thiếu, mặc định cửa hàng mở cửa
    df['Open'].fillna(1, inplace=True)


## TẠO CÁC ĐẶC TRƯNG

# Tạo đặc trưng thời gian từ cột Date
for df in [train, test]:
    # Chuyển cột Date sang kiểu thời gian
    df['Date'] = pd.to_datetime(df['Date'])
    # Tách các thành phần thời gian
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    df['IsWeekend'] = df['DayOfWeek'].isin([6,7]).astype(int)



train[['Date','Year','Month','Day','WeekOfYear','Quarter','IsWeekend']].head()


# Tạo đặc trưng nâng cao
months = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
          "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

for df in [train, test]:
    # Đang khuyến mãi mở rộng (Promo2Active)
    def promo2_active(p2, interval, month):
        if p2 == 0 or interval == 'None': return 0
        return int(month in [months[m] for m in interval.split(',') if m in months])
    df['Promo2Active'] = df.apply(lambda r: promo2_active(r['Promo2'], r['PromoInterval'], r['Month']), axis=1)
    #  Có đối thủ cạnh tranh gần
    df['CompetitionActive'] = (df['CompetitionDistance'] < 10000).astype(int)
    # Số tháng kể từ khi có đối thủ
    df['CompetitionMonths'] = ((df['Year'] - df['CompetitionOpenSinceYear']) * 12 +
                               (df['Month'] - df['CompetitionOpenSinceMonth'])).clip(lower=0).fillna(0).astype(int)


# Số tuần kể từ khi tham gia Promo2
import numpy as np
for df in [train, test]:
    # Nếu cửa hàng không có Promo2 thì gán 0
    df['Promo2Weeks'] = np.where(
        df['Promo2'] == 1,
        ((df['Year'] - df['Promo2SinceYear']) * 52 +
         (df['WeekOfYear'] - df['Promo2SinceWeek'])).clip(lower=0).fillna(0),
        0
    ).astype(int)

train[['Store', 'Promo2Active', 'CompetitionActive', 'CompetitionMonths', 'Promo2Weeks']].head()


## MÃ HÓA CÁC CỘT

# Mã hóa (Encoding)
from sklearn.preprocessing import LabelEncoder

# Các cột dạng chuỗi cần mã hóa
cat_cols = ['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval']

# Mã hóa từng cột
le = LabelEncoder()
for col in cat_cols:
    train[col] = train[col].astype(str).fillna("None")
    test[col]  = test[col].astype(str).fillna("None")

    all_values = pd.concat([train[col], test[col]])
    le.fit(all_values)
    train[col] = le.transform(train[col])
    test[col]  = le.transform(test[col])


train[['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval']].head()


import numpy as np

# Loại các dòng không cần thiết
train = train[train['Open'] == 1].copy()
train['Sales_log1p'] = np.log1p(train['Sales'])

# Tách dữ liệu đầu vào (X) và đầu ra (y)
X_train = train.drop(columns=['Sales', 'Sales_log1p', 'Date'])
y_train = train['Sales_log1p']
X_test = test.drop(columns=['Date'], errors='ignore')

print("Dữ liệu huấn luyện:")
print("X_train:", X_train.shape, " | y_train:", y_train.shape)
print("X_test:", X_test.shape)


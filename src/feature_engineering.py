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

# Chọn các đặc trưng
from sklearn.model_selection import train_test_split

features = [
    'Store', 'Promo', 'CompetitionDistance', 'CompetitionActive', 'CompetitionMonths',
    'Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3',
    'Month', 'Quarter', 'Year', 'IsWeekend', 'IsHolidaySeason', 'Customers', 'DaysAhead'
]

X = df[features]
y = df['Sales']

# HUẤN LUYỆN MÔ HÌNH

## RANDOM FOREST

!rm -f *.pkl
rf = None


import pandas as pd
import numpy as np

train = pd.read_csv(f"https://drive.google.com/uc?id={file_ids['train']}")
store = pd.read_csv(f"https://drive.google.com/uc?id={file_ids['store']}")


df = pd.merge(train, store, on='Store', how='left')

# Chuyển cột Date về dạng datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Store', 'Date'])


# các đặc trưng thời gian
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['IsWeekend'] = df['Date'].dt.dayofweek.isin([5,6]).astype(int)
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['IsHolidaySeason'] = df['Month'].isin([11, 12]).astype(int)

df['DaysAhead'] = (df['Date'] - df['Date'].min()).dt.days



for lag in [1, 2, 3]:
    df[f'Sales_Lag{lag}'] = df.groupby('Store')['Sales'].shift(lag)

df = df.dropna()


print(df.columns.tolist())


import numpy as np
import pandas as pd

df['Date'] = pd.to_datetime(df['Date'])
df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(2013).astype(int)
df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(1).astype(int)
df['CompetitionActive'] = np.where(
    (df['CompetitionOpenSinceYear'] < df['Year']) |
    ((df['CompetitionOpenSinceYear'] == df['Year']) &
     (df['CompetitionOpenSinceMonth'] <= df['Month'])),
    1, 0
)

df['CompetitionMonths'] = (
    12 * (df['Year'] - df['CompetitionOpenSinceYear'])
    + (df['Month'] - df['CompetitionOpenSinceMonth'])
)
df['CompetitionMonths'] = df['CompetitionMonths'].clip(lower=0)


from sklearn.model_selection import train_test_split

features = [
    'Store', 'Promo', 'CompetitionDistance', 'CompetitionActive', 'CompetitionMonths',
    'Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3',
    'Month', 'Quarter', 'Year', 'IsWeekend', 'IsHolidaySeason', 'Customers', 'DaysAhead'
]

X = df[features]
y = df['Sales']

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print("RMSE:", rmse)
print("R²:", r2)


## LIGHTGBM

!pip install lightgbm


from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

lgbm = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgbm.fit(X_train, y_train)

y_pred_lgbm = lgbm.predict(X_val)

rmse_lgbm = np.sqrt(mean_squared_error(y_val, y_pred_lgbm))
r2_lgbm = r2_score(y_val, y_pred_lgbm)

print(f"RMSE: {rmse_lgbm:.4f}")
print(f"R²  : {r2_lgbm:.4f}")


import joblib
joblib.dump(lgbm, "lightgbm_sales_model.pkl")

## LINEAR REGRESSION

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

features = [
    'Store', 'Promo', 'CompetitionDistance', 'CompetitionActive', 'CompetitionMonths',
    'Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3',
    'Month', 'Quarter', 'Year', 'IsWeekend', 'IsHolidaySeason', 'Customers', 'DaysAhead'
]

X_train_lr = X_train[features].copy()
X_val_lr   = X_val[features].copy()

imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_lr)
X_val_imputed   = imputer.transform(X_val_lr)

lr = LinearRegression(n_jobs=-1)
lr.fit(X_train_imputed, y_train)

y_pred_lr = lr.predict(X_val_imputed)
rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))
r2_lr = r2_score(y_val, y_pred_lr)

print(f"RMSE: {rmse_lr:.4f}")
print(f"R²: {r2_lr:.4f}")


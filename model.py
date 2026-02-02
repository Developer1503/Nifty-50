# ===============================
# NIFTY 50 Stock Price Prediction
# Ridge Regression (Time-Series)
# ===============================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("dataset/NIFTY50_all.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Filter one stock (RELIANCE)
df = df[df["Symbol"] == "RELIANCE"].copy()
df.reset_index(drop=True, inplace=True)

# -------------------------------
# 2. Feature Engineering
# -------------------------------

# Returns & Momentum
df["Return"] = df["Close"].pct_change()
df["Momentum_4"] = df["Close"] - df["Close"].shift(4)

# Moving Averages
df["MA_7"] = df["Close"].rolling(7).mean()
df["MA_21"] = df["Close"].rolling(21).mean()

# Volatility
df["Volatility_7"] = df["Close"].rolling(7).std()

# Lag Features
df["Close_Lag1"] = df["Close"].shift(1)
df["Close_Lag2"] = df["Close"].shift(2)
df["Close_Lag3"] = df["Close"].shift(3)

# Target Variable (Next-day Close)
df["Target"] = df["Close"].shift(-1)

# Drop rows with NaN values
df.dropna(inplace=True)

# -------------------------------
# 3. Feature & Label Selection
# -------------------------------
features = [
    "Open", "High", "Low", "Close", "Volume",
    "Return", "Momentum_4",
    "MA_7", "MA_21",
    "Volatility_7",
    "Close_Lag1", "Close_Lag2", "Close_Lag3"
]

X = df[features]
y = df["Target"]

# -------------------------------
# 4. Time-Series Train/Test Split
# -------------------------------
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# -------------------------------
# 5. Feature Scaling (Train Only)
# -------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 6. Train Ridge Regression Model
# -------------------------------
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# -------------------------------
# 7. Predictions
# -------------------------------
y_pred = model.predict(X_test_scaled)

# -------------------------------
# 8. Evaluation
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# -------------------------------
# 9. Predict Next-Day Price
# -------------------------------
latest_data = X.iloc[[-1]]
latest_scaled = scaler.transform(latest_data)

tomorrow_price = model.predict(latest_scaled)[0]
print(f"\nPredicted next-day Close price: ₹{tomorrow_price:.2f}")

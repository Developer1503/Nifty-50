# ============================================================
# Multi-Stock ML Portfolio Strategy (NIFTY 50)
# - Direction Classification
# - Volatility Filter
# - Walk-forward Retraining
# - Equal-weight Portfolio
# - Transaction Costs + Sharpe
# ============================================================

import kagglehub
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------------------------
# 1. Load Dataset
# ---------------------------
path = kagglehub.dataset_download("rohanrao/nifty50-stock-market-data")
df_all = pd.read_csv(f"{path}/NIFTY50_all.csv")

df_all["Date"] = pd.to_datetime(df_all["Date"])
df_all = df_all.sort_values("Date")

# Stocks for portfolio
STOCKS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

TRANSACTION_COST = 0.001  # 0.1%

portfolio_returns = []

# ---------------------------
# 2. Loop Over Stocks
# ---------------------------
for stock in STOCKS:

    df = df_all[df_all["Symbol"] == stock].copy()
    df.reset_index(drop=True, inplace=True)

    # Feature engineering
    df["Return"] = df["Close"].pct_change()
    df["Momentum_4"] = df["Close"] - df["Close"].shift(4)

    df["MA_7"] = df["Close"].rolling(7).mean()
    df["MA_21"] = df["Close"].rolling(21).mean()

    df["Volatility_7"] = df["Return"].rolling(7).std()

    df["Close_Lag1"] = df["Close"].shift(1)
    df["Close_Lag2"] = df["Close"].shift(2)
    df["Close_Lag3"] = df["Close"].shift(3)

    df["Target_Direction"] = (df["Return"].shift(-1) > 0).astype(int)

    df.dropna(inplace=True)

    features = [
        "Return", "Momentum_4",
        "MA_7", "MA_21",
        "Volatility_7",
        "Close_Lag1", "Close_Lag2", "Close_Lag3"
    ]

    # Walk-forward parameters
    train_size = int(len(df) * 0.6)
    test_size = int(len(df) * 0.1)

    vol_threshold = df["Volatility_7"].median()
    stock_returns = []

    # Walk-forward loop
    for start in range(0, len(df) - train_size - test_size, test_size):

        train = df.iloc[start:start + train_size]
        test  = df.iloc[start + train_size:start + train_size + test_size]

        X_train = train[features]
        y_train = train["Target_Direction"]

        X_test = test[features]
        y_test = test["Return"].shift(-1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

        prob_up = model.predict_proba(X_test_scaled)[:, 1]

        trade_allowed = test["Volatility_7"].values < vol_threshold

        signals = np.where(
            (prob_up > 0.55) & trade_allowed,  1,
            np.where((prob_up < 0.45) & trade_allowed, -1, 0)
        )

        daily_returns = signals * y_test.values
        costs = np.abs(signals) * TRANSACTION_COST
        daily_returns = daily_returns - costs

        stock_returns.extend(daily_returns)

    portfolio_returns.append(pd.Series(stock_returns))

# ---------------------------
# 3. Portfolio Aggregation
# ---------------------------
portfolio_df = pd.concat(portfolio_returns, axis=1)
portfolio_df.columns = STOCKS

# Equal-weight portfolio
portfolio_daily_returns = portfolio_df.mean(axis=1)

# ---------------------------
# 4. Portfolio Metrics
# ---------------------------
portfolio_daily_returns = portfolio_daily_returns.dropna()

total_return = np.prod(1 + portfolio_daily_returns) - 1
annual_volatility = portfolio_daily_returns.std() * np.sqrt(252)
sharpe_ratio = (
    portfolio_daily_returns.mean()
    / portfolio_daily_returns.std()
    * np.sqrt(252)
)

print("📈 MULTI-STOCK PORTFOLIO RESULTS")
print(f"Total Return        : {total_return*100:.2f}%")
print(f"Annual Volatility   : {annual_volatility*100:.2f}%")
print(f"Sharpe Ratio        : {sharpe_ratio:.2f}")

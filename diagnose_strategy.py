"""
Strategy Diagnostic Tool
Analyzes why the current strategy has Sharpe = -0.03
"""

import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("STRATEGY DIAGNOSTIC ANALYSIS")
print("="*80)

# Load data
print("\nLoading data...")
path = kagglehub.dataset_download("rohanrao/nifty50-stock-market-data")
df_all = pd.read_csv(f"{path}/NIFTY50_all.csv")
df_all["Date"] = pd.to_datetime(df_all["Date"])
df_all = df_all.sort_values("Date")

# Analyze one stock in detail
STOCK = "RELIANCE"
df = df_all[df_all["Symbol"] == STOCK].copy()
df.reset_index(drop=True, inplace=True)

print(f"\n📈 Analyzing {STOCK}")
print(f"   Data points: {len(df)}")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

# Feature engineering (same as original)
df["Return"] = df["Close"].pct_change()
df["Momentum_4"] = df["Close"] - df["Close"].shift(4)
df["MA_7"] = df["Close"].rolling(7).mean()
df["MA_21"] = df["Close"].rolling(21).mean()
df["Volatility_7"] = df["Return"].rolling(7).std()
df["Close_Lag1"] = df["Close"].shift(1)
df["Close_Lag2"] = df["Close"].shift(2)
df["Close_Lag3"] = df["Close"].shift(3)
df["Target_Direction"] = (df["Return"].shift(-1) > 0).astype(int)
df["Target_Return"] = df["Return"].shift(-1)
df.dropna(inplace=True)

features = [
    "Return", "Momentum_4",
    "MA_7", "MA_21",
    "Volatility_7",
    "Close_Lag1", "Close_Lag2", "Close_Lag3"
]

# Split data
train_size = int(len(df) * 0.6)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

print(f"\n📊 Data Split:")
print(f"   Training: {len(train)} samples")
print(f"   Testing:  {len(test)} samples")

# Train model
X_train = train[features]
y_train = train["Target_Direction"]
X_test = test[features]
y_test_direction = test["Target_Direction"]
y_test_return = test["Target_Return"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predictions
probs = model.predict_proba(X_test_scaled)[:, 1]
predictions = (probs > 0.5).astype(int)

# ============================================================
# DIAGNOSTIC 1: Model Accuracy
# ============================================================
print("\n" + "="*80)
print("1️⃣  MODEL PREDICTION ACCURACY")
print("="*80)

accuracy = accuracy_score(y_test_direction, predictions)
market_up_rate = (y_test_direction == 1).sum() / len(y_test_direction)

print(f"\nModel Accuracy:        {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"Market Up Rate:        {market_up_rate:.3f} ({market_up_rate*100:.1f}%)")
print(f"Random Baseline:       0.500 (50.0%)")
print(f"Improvement over random: {(accuracy - 0.5)*100:.1f}%")

if accuracy < 0.52:
    print("\n⚠️  CRITICAL ISSUE: Model is barely better than random!")
    print("   → The model cannot reliably predict price direction")
elif accuracy < 0.55:
    print("\n⚠️  WARNING: Model accuracy is marginal")
    print("   → Small edge may be eroded by transaction costs")
else:
    print("\n✅ Model shows predictive power")

# Confusion Matrix
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test_direction, predictions)
print(f"                Predicted")
print(f"              Down    Up")
print(f"Actual Down   {cm[0,0]:<6}  {cm[0,1]:<6}")
print(f"       Up     {cm[1,0]:<6}  {cm[1,1]:<6}")

# ============================================================
# DIAGNOSTIC 2: Probability Calibration
# ============================================================
print("\n" + "="*80)
print("2️⃣  PROBABILITY CALIBRATION")
print("="*80)

bins = [0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0]
test_df = test.copy()
test_df['Prob'] = probs
test_df['Prob_Bin'] = pd.cut(probs, bins=bins)

calibration = test_df.groupby('Prob_Bin', observed=True).agg({
    'Target_Direction': ['mean', 'count']
})

print("\nPredicted Probability → Actual Win Rate:")
print("Prob Range    Actual Win Rate    Count")
print("-" * 45)
for idx, row in calibration.iterrows():
    actual_rate = row[('Target_Direction', 'mean')]
    count = int(row[('Target_Direction', 'count')])
    print(f"{str(idx):<12}  {actual_rate:>6.1%}            {count:>5}")

print("\n💡 Interpretation:")
print("   If model is well-calibrated, 60% probability → ~60% actual win rate")

# ============================================================
# DIAGNOSTIC 3: Trading Signals
# ============================================================
print("\n" + "="*80)
print("3️⃣  TRADING SIGNAL ANALYSIS")
print("="*80)

vol_threshold = test["Volatility_7"].median()
trade_allowed = test["Volatility_7"].values < vol_threshold

signals = np.where(
    (probs > 0.55) & trade_allowed, 1,
    np.where((probs < 0.45) & trade_allowed, -1, 0)
)

n_long = (signals == 1).sum()
n_short = (signals == -1).sum()
n_flat = (signals == 0).sum()
total = len(signals)

print(f"\nSignal Distribution:")
print(f"  Long (Buy):     {n_long:>5} ({n_long/total*100:>5.1f}%)")
print(f"  Short (Sell):   {n_short:>5} ({n_short/total*100:>5.1f}%)")
print(f"  Flat (No trade): {n_flat:>5} ({n_flat/total*100:>5.1f}%)")

if n_flat > 0.7 * total:
    print("\n⚠️  WARNING: Too many 'no trade' signals!")
    print("   → Thresholds (0.55/0.45) may be too conservative")
    print("   → Missing trading opportunities")

# ============================================================
# DIAGNOSTIC 4: Returns by Signal Type
# ============================================================
print("\n" + "="*80)
print("4️⃣  RETURNS BY SIGNAL TYPE")
print("="*80)

test_df['Signal'] = signals
signal_analysis = test_df.groupby('Signal')['Target_Return'].agg([
    ('Mean_Return', 'mean'),
    ('Std', 'std'),
    ('Count', 'count'),
    ('Win_Rate', lambda x: (x > 0).mean()),
    ('Total_Return', 'sum')
])

print("\nSignal  Mean Return   Std Dev   Count   Win Rate   Total Return")
print("-" * 70)
for signal_type in [-1, 0, 1]:
    if signal_type in signal_analysis.index:
        row = signal_analysis.loc[signal_type]
        signal_name = {-1: "Short", 0: "Flat", 1: "Long"}[signal_type]
        print(f"{signal_name:<6}  {row['Mean_Return']*100:>6.3f}%    "
              f"{row['Std']*100:>6.3f}%  {int(row['Count']):>5}   "
              f"{row['Win_Rate']:>6.1%}     {row['Total_Return']*100:>6.2f}%")

print("\n💡 Key Questions:")
print("   • Are long signals profitable on average?")
print("   • Are short signals profitable on average?")
print("   • Is the win rate > 50% for directional trades?")

# ============================================================
# DIAGNOSTIC 5: Transaction Costs Impact
# ============================================================
print("\n" + "="*80)
print("5️⃣  TRANSACTION COSTS IMPACT")
print("="*80)

TRANSACTION_COST = 0.001  # 0.1%

strategy_returns = signals * y_test_return.values
costs = np.abs(signals) * TRANSACTION_COST

gross_return = strategy_returns.sum()
total_costs = costs.sum()
net_return = gross_return - total_costs

print(f"\nGross Returns (before costs):  {gross_return*100:>8.2f}%")
print(f"Transaction Costs:             {total_costs*100:>8.2f}%")
print(f"Net Returns (after costs):     {net_return*100:>8.2f}%")

if gross_return != 0:
    cost_impact = (total_costs / abs(gross_return)) * 100
    print(f"\nCosts consume {cost_impact:.1f}% of gross returns!")
    
    if cost_impact > 50:
        print("\n⚠️  CRITICAL: Transaction costs eating >50% of profits!")
        print("   → Need to reduce trade frequency")
        print("   → Consider stricter thresholds or longer holding periods")

# ============================================================
# DIAGNOSTIC 6: Volatility Filter Effectiveness
# ============================================================
print("\n" + "="*80)
print("6️⃣  VOLATILITY FILTER EFFECTIVENESS")
print("="*80)

high_vol_mask = ~trade_allowed
low_vol_mask = trade_allowed

high_vol_returns = test[high_vol_mask]['Target_Return'].mean()
low_vol_returns = test[low_vol_mask]['Target_Return'].mean()
high_vol_count = high_vol_mask.sum()
low_vol_count = low_vol_mask.sum()

print(f"\nHigh Volatility Periods:")
print(f"  Count:        {high_vol_count}")
print(f"  Avg Return:   {high_vol_returns*100:.4f}%")

print(f"\nLow Volatility Periods:")
print(f"  Count:        {low_vol_count}")
print(f"  Avg Return:   {low_vol_returns*100:.4f}%")

if high_vol_returns > low_vol_returns:
    print("\n⚠️  ISSUE: Volatility filter is BACKWARDS!")
    print("   → High volatility periods are MORE profitable")
    print("   → Filter is blocking good trading opportunities")
else:
    print("\n✅ Volatility filter is working as intended")

# ============================================================
# DIAGNOSTIC 7: Feature Importance
# ============================================================
print("\n" + "="*80)
print("7️⃣  FEATURE IMPORTANCE (Logistic Regression Coefficients)")
print("="*80)

coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
})
coefficients['Abs_Coef'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values('Abs_Coef', ascending=False)

print("\nFeature              Coefficient    Importance")
print("-" * 50)
for _, row in coefficients.iterrows():
    print(f"{row['Feature']:<20} {row['Coefficient']:>10.4f}    {'█' * int(row['Abs_Coef'] * 20)}")

# ============================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================
print("\n" + "="*80)
print("📋 SUMMARY & RECOMMENDATIONS")
print("="*80)

issues = []
recommendations = []

if accuracy < 0.52:
    issues.append("Model accuracy too low (barely better than random)")
    recommendations.append("Add more predictive features: RSI, MACD, Bollinger Bands, volume indicators")
    recommendations.append("Try more sophisticated models: Random Forest, XGBoost, LSTM")

if n_flat > 0.7 * total:
    issues.append("Too many 'no trade' signals (>70%)")
    recommendations.append("Relax thresholds from 0.55/0.45 to 0.52/0.48")
    recommendations.append("Or use continuous position sizing instead of binary signals")

if gross_return != 0 and (total_costs / abs(gross_return)) > 0.5:
    issues.append("Transaction costs consuming >50% of gross returns")
    recommendations.append("Reduce trade frequency with stricter thresholds")
    recommendations.append("Implement minimum holding period")
    recommendations.append("Consider lower-cost execution methods")

if high_vol_returns > low_vol_returns:
    issues.append("Volatility filter is counterproductive")
    recommendations.append("Remove or reverse the volatility filter")
    recommendations.append("High volatility = more opportunity, not more risk")

if net_return < 0:
    issues.append("Strategy has negative returns")
    recommendations.append("Consider long-only strategy (remove short signals)")
    recommendations.append("Market has natural upward drift - shorting is harder")

print("\n🔴 IDENTIFIED ISSUES:")
for i, issue in enumerate(issues, 1):
    print(f"   {i}. {issue}")

print("\n💡 RECOMMENDED ACTIONS:")
for i, rec in enumerate(recommendations, 1):
    print(f"   {i}. {rec}")

print("\n" + "="*80)
print("✅ DIAGNOSTIC COMPLETE")
print("="*80)

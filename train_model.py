
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
DATASET_PATH = 'c:/ML-aiproject/Nifty-50/dataset/NIFTY50_all.csv'
STOCK_SYMBOL = 'RELIANCE'  # You can change this to TCS, HDFCBANK, etc.
PREDICTION_DAYS = 1        # Predict 'Current Day' Close or 'Next Day' (using shift)

def train_stock_model():
    print(f"--- Training Model for {STOCK_SYMBOL} ---\n")

    # 1. Load Data
    try:
        df = pd.read_csv(DATASET_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        # Filter for specific stock
        data = df[df['Symbol'] == STOCK_SYMBOL].sort_values('Date').copy()
        
        if data.empty:
            print(f"No data found for symbol {STOCK_SYMBOL}")
            return
            
        print(f"Loaded {len(data)} records for {STOCK_SYMBOL} from {data['Date'].min().date()} to {data['Date'].max().date()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Feature Engineering
    # We will use past data to predict the future (or current Close based on Open/High/Low if purely intraday, 
    # but usually we want to predict *next day* based on *past days* or *current Close* based on *engineered features*)
    
    # Let's create Lag features (Past 3 days closing price)
    data['Close_Lag1'] = data['Close'].shift(1)
    data['Close_Lag2'] = data['Close'].shift(2)
    data['Close_Lag3'] = data['Close'].shift(3)
    
    # Moving Averages
    data['MA_7'] = data['Close'].rolling(window=7).mean()
    data['MA_21'] = data['Close'].rolling(window=21).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    
    # Volatility
    data['Volatility'] = data['Close'].rolling(window=7).std()

    # Drop rows with NaN created by shifting/rolling
    data = data.dropna()

    # Define Features (X) and Target (y)
    # Goal: Predict 'Close' price of today based on Past Days data? 
    # OR Predict 'Next Day Close' based on Today's data?
    # Let's predict *Tomorrow's Close* (Target) using *Today's Features*
    data['Target'] = data['Close'].shift(-1)
    data = data.dropna()

    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'Close_Lag1', 'Close_Lag2', 'MA_7', 'MA_21', 'MA_50', 'Volatility']
    X = data[features]
    y = data['Target']

    # 3. Time-Series Split (Don't shuffle!)
    # We must split by time: Train on past, Test on future.
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    test_dates = data['Date'].iloc[train_size:]

    print(f"\nTraining Set: {X_train.shape[0]} samples")
    print(f"Testing Set:  {X_test.shape[0]} samples")

    # 4. Train Model
    print("\nTraining Linear Regression Model...")
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Evaluate
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.2f}")
    print(f"R² Score: {r2:.4f} (1.0 is perfect)")

    # 6. Coefficients (Feature Importance for Linear Model)
    print("\n--- Feature Importance (Coefficients) ---")
    coeffs = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_}).sort_values('Coefficient', ascending=False)
    print(coeffs.head(5))

    # Optional: Save a plot if user validates
    # (Leaving out graphical plotting pop-up to keep it headless, but we can save stats)

if __name__ == "__main__":
    train_stock_model()

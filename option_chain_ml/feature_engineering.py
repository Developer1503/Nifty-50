"""
feature_engineering.py
=======================
Builds option-chain-style features from NIFTY50 OHLCV data.

Since we have historical stock-level OHLCV data (not a live option chain feed),
this module:
  1. Reconstructs a "synthetic" NIFTY spot index by averaging Nifty-50 constituents.
  2. Simulates option-chain features that a real chain would provide:
       - Put-Call Ratio (PCR) via volume asymmetry proxy
       - Implied Volatility (IV) via realised-vol expansion
       - OI Concentration (max OI strike) proxy
       - Max Pain level (strike with least OI-weighted payout)
  3. Creates standard technical features (RSI, MACD, Bollinger Bands, ATR, VWAP).
  4. Produces the final label: 3-class direction over the next N days.

All functions are pure (no side-effects) and return DataFrames / Series.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

# ── Constants ───────────────────────────────────────────────────────────────
NIFTY50_SYMBOLS = [
    "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV",
    "BAJFINANCE", "BHARTIARTL", "BPCL", "BRITANNIA", "CIPLA",
    "COALINDIA", "DRREDDY", "EICHERMOT", "GAIL", "GRASIM",
    "HCLTECH", "HDFC", "HDFCBANK", "HEROMOTOCO", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "IOC",
    "ITC", "JSWSTEEL", "KOTAKBANK", "LT", "MARUTI",
    "MM", "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
    "RELIANCE", "SBIN", "SHREECEM", "SUNPHARMA", "TATAMOTORS",
    "TATASTEEL", "TCS", "TECHM", "TITAN", "ULTRACEMCO",
    "UPL", "VEDL", "WIPRO", "ZEEL",
]

FORWARD_DAYS = 3           # prediction horizon
SIDEWAYS_BAND = 0.005      # ±0.5% → sideways label


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_nifty50_all(filepath: Path | None = None) -> pd.DataFrame:
    """Load the master NIFTY50 CSV (all stocks combined)."""
    fp = filepath or DATASET_DIR / "NIFTY50_all.csv"
    df = pd.read_csv(fp, parse_dates=["Date"], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # Keep only EQ series
    df = df[df["Series"] == "EQ"].copy()
    df = df[df["Symbol"].isin(NIFTY50_SYMBOLS)].copy()
    df["Close"]   = pd.to_numeric(df["Close"],   errors="coerce")
    df["Volume"]  = pd.to_numeric(df["Volume"],  errors="coerce")
    df["Open"]    = pd.to_numeric(df["Open"],    errors="coerce")
    df["High"]    = pd.to_numeric(df["High"],    errors="coerce")
    df["Low"]     = pd.to_numeric(df["Low"],     errors="coerce")
    df["VWAP"]    = pd.to_numeric(df["VWAP"],    errors="coerce")
    df = df.dropna(subset=["Close", "Volume"])
    return df.sort_values(["Date", "Symbol"]).reset_index(drop=True)


def build_nifty_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a synthetic NIFTY spot index from constituent closes.
    Uses equal-weight average (real NIFTY is free-float market-cap weighted,
    but equal-weight is a good proxy for feature engineering).
    """
    pivot = (
        df.pivot_table(index="Date", columns="Symbol", values="Close", aggfunc="last")
        .sort_index()
    )
    # Drop stocks with <80% coverage
    coverage = pivot.notna().mean()
    pivot = pivot[coverage[coverage >= 0.80].index]
    # Normalise each stock to its first valid price so they are on the same scale
    norm = pivot / pivot.bfill().iloc[0]
    index_series = norm.mean(axis=1) * 10000          # scale to ~Nifty level
    index_df = pd.DataFrame({
        "Date":  index_series.index,
        "Spot":  index_series.values,
    }).reset_index(drop=True)

    # Add OHLCV aggregated columns from the full dataset
    agg = df.groupby("Date").agg(
        Volume=("Volume", "sum"),
        High_avg=("High",   "mean"),
        Low_avg=("Low",    "mean"),
        Open_avg=("Open",  "mean"),
        Close_avg=("Close","mean"),
        VWAP_avg=("VWAP",  "mean"),
    ).reset_index()

    index_df = index_df.merge(agg, on="Date", how="left")
    return index_df


# ═══════════════════════════════════════════════════════════════════════════
# 2.  OPTION-CHAIN FEATURE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def simulate_option_chain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive option-chain proxies from price & volume data.

    PCR (Put-Call Ratio):
        Estimated by volume-asymmetry: when volume surges on down-days it
        mirrors elevated put buying; the opposite for calls.  We compute a
        smoothed 5-day ratio.

    IV (Implied Volatility proxy):
        21-day realised vol annualised.  High IV → fear / uncertainty.

    OI Concentration:
        ATR as a fraction of spot price measures how "clustered" the price
        action is.  Tight ATR → OI concentrated near current level.

    Max Pain:
        The strike with minimum aggregate payout to option holders. We
        approximate it as the rolling 10-day VWAP (options market-makers
        pin price near the max-pain level in the last week of expiry).

    Returns df with new columns added.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)
    spot = df["Spot"]
    ret  = spot.pct_change()

    # ── PCR proxy ─────────────────────────────────────────────────────────
    # Volume on down-days ≈ put OI proxy; up-days ≈ call OI proxy
    down_vol = df["Volume"].where(ret < 0, 0)
    up_vol   = df["Volume"].where(ret > 0, 0)
    pcr_raw  = (
        down_vol.rolling(5, min_periods=1).sum() /
        (up_vol.rolling(5, min_periods=1).sum() + 1e-9)
    )
    df["PCR"] = pcr_raw.clip(0, 5)         # cap at 5 to remove outliers

    # ── IV proxy (realised vol, 21-day) ───────────────────────────────────
    df["IV"] = ret.rolling(21, min_periods=5).std() * np.sqrt(252) * 100   # %

    # ── OI Concentration (ATR-based) ──────────────────────────────────────
    high_low_range = df["High_avg"] - df["Low_avg"]
    atr = high_low_range.rolling(14, min_periods=3).mean()
    df["OI_Concentration"] = 1 - (atr / spot).clip(0, 1)  # higher → more concentrated

    # ── Max Pain proxy (10-day VWAP) ──────────────────────────────────────
    df["Max_Pain"] = df["VWAP_avg"].rolling(10, min_periods=3).mean()
    # Distance from spot to max pain (normalised)
    df["Max_Pain_Distance"] = (spot - df["Max_Pain"]) / spot

    # ── PCR change (momentum of PCR) ──────────────────────────────────────
    df["PCR_Change"] = df["PCR"].pct_change(3)

    # ── IV change ─────────────────────────────────────────────────────────
    df["IV_Change"] = df["IV"].pct_change(5)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3.  TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add standard TA features to the index DataFrame."""
    df = df.copy()
    c = df["Spot"]
    ret = c.pct_change()

    # ── Returns ───────────────────────────────────────────────────────────
    df["Ret_1"]  = ret
    df["Ret_3"]  = c.pct_change(3)
    df["Ret_5"]  = c.pct_change(5)
    df["Ret_10"] = c.pct_change(10)
    df["Ret_20"] = c.pct_change(20)

    # ── Moving Averages ───────────────────────────────────────────────────
    for w in [5, 10, 20, 50]:
        df[f"MA_{w}"]       = c.rolling(w).mean()
        df[f"MA_dist_{w}"]  = (c - df[f"MA_{w}"]) / df[f"MA_{w}"]   # normalised dist

    # ── EMA ───────────────────────────────────────────────────────────────
    df["EMA_12"] = c.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = c.ewm(span=26, adjust=False).mean()

    # ── MACD ──────────────────────────────────────────────────────────────
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ── RSI (14-period) ───────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── Stochastic RSI ────────────────────────────────────────────────────
    rsi_min = df["RSI"].rolling(14).min()
    rsi_max = df["RSI"].rolling(14).max()
    df["StochRSI"] = (df["RSI"] - rsi_min) / (rsi_max - rsi_min + 1e-9)

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["BB_Upper"]    = bb_mid + 2 * bb_std
    df["BB_Lower"]    = bb_mid - 2 * bb_std
    df["BB_Position"] = (c - bb_mid) / (2 * bb_std + 1e-9)   # -1..1
    df["BB_Width"]    = (df["BB_Upper"] - df["BB_Lower"]) / (bb_mid + 1e-9)

    # ── ATR (14-period) ───────────────────────────────────────────────────
    high = df["High_avg"]
    low  = df["Low_avg"]
    tr   = pd.concat([high - low,
                      (high - c.shift()).abs(),
                      (low  - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"]     = tr.rolling(14).mean()
    df["ATR_pct"] = df["ATR"] / c * 100

    # ── Volatility ratios ─────────────────────────────────────────────────
    vol_21 = ret.rolling(21).std() * np.sqrt(252)
    vol_5  = ret.rolling(5).std()  * np.sqrt(252)
    df["Vol_5_21_Ratio"] = vol_5 / (vol_21 + 1e-9)   # > 1 → vol expanding

    # ── Volume features ───────────────────────────────────────────────────
    df["Vol_MA5"]   = df["Volume"].rolling(5).mean()
    df["Vol_Ratio"] = df["Volume"] / (df["Vol_MA5"] + 1e-9)  # > 1 → spike

    # ── Trend strength (ADX proxy) ────────────────────────────────────────
    df["Trend_Strength"] = df["MA_dist_20"].abs()   # simple momentum magnitude

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4.  TOP GAINERS / LOSERS BREADTH FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def add_breadth_features(index_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market breadth from constituent stock returns.

    Features added:
        - ADR:                   Advance / Decline Ratio on each date
        - Breadth_Net:           (gainers - losers) / total stocks
        - Top_Gainer_Ret:        mean return of top-5 gainers
        - Top_Loser_Ret:         mean return of top-5 losers (positive magnitude)
        - Gainer_Loser_Spread:   top_gainer_ret - top_loser_ret
    """
    # Build daily returns per symbol per date
    tmp = raw_df[["Date", "Symbol", "Close"]].copy()
    tmp = tmp.sort_values(["Symbol", "Date"])
    tmp["DailyRet"] = tmp.groupby("Symbol")["Close"].pct_change()
    tmp = tmp.dropna(subset=["DailyRet"])

    # ── Vectorised breadth metrics ────────────────────────────────────────
    g = tmp.groupby("Date")["DailyRet"]

    gainers = tmp.assign(is_up=(tmp["DailyRet"] > 0).astype(int)) \
                 .groupby("Date")["is_up"].sum().rename("gainers")
    losers  = tmp.assign(is_dn=(tmp["DailyRet"] < 0).astype(int)) \
                 .groupby("Date")["is_dn"].sum().rename("losers")
    totals  = tmp.groupby("Date")["DailyRet"].count().rename("total")

    breadth = pd.concat([gainers, losers, totals], axis=1)
    breadth["ADR"]         = breadth["gainers"] / (breadth["losers"] + 1e-9)
    breadth["Breadth_Net"] = (breadth["gainers"] - breadth["losers"]) / (
        breadth["total"] + 1e-9)

    # Top-5 gainer / loser returns per date
    def top5_mean_gain(series):
        top = series.nlargest(5)
        return top.mean() if len(top) > 0 else 0.0

    def top5_mean_loss(series):
        bot = series.nsmallest(5).abs()
        return bot.mean() if len(bot) > 0 else 0.0

    breadth["Top_Gainer_Ret"] = g.apply(top5_mean_gain)
    breadth["Top_Loser_Ret"]  = g.apply(top5_mean_loss)
    breadth["Gainer_Loser_Spread"] = (
        breadth["Top_Gainer_Ret"] - breadth["Top_Loser_Ret"]
    )

    breadth = breadth[["ADR", "Breadth_Net", "Top_Gainer_Ret",
                        "Top_Loser_Ret", "Gainer_Loser_Spread"]].reset_index()

    index_df = index_df.merge(breadth, on="Date", how="left")

    # Smooth breadth with 3-day MA
    for col in ["ADR", "Breadth_Net", "Gainer_Loser_Spread"]:
        index_df[f"{col}_MA3"] = index_df[col].rolling(3).mean()

    return index_df


# ═══════════════════════════════════════════════════════════════════════════
# 5.  TARGET VARIABLE
# ═══════════════════════════════════════════════════════════════════════════

def add_target(df: pd.DataFrame, forward_days: int = FORWARD_DAYS,
               band: float = SIDEWAYS_BAND) -> pd.DataFrame:
    """
    3-class label:
        0 → Down    (forward return < -band)
        1 → Sideways (|forward return| ≤ band)
        2 → Up      (forward return > +band)
    """
    fwd_ret = df["Spot"].pct_change(forward_days).shift(-forward_days)

    conditions = [
        fwd_ret < -band,
        fwd_ret >  band,
    ]
    df["Target"]     = np.select(conditions, [0, 2], default=1)
    df["Fwd_Return"] = fwd_ret
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 6.  MASTER PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    # Option-chain proxies
    "PCR", "PCR_Change", "IV", "IV_Change",
    "OI_Concentration", "Max_Pain_Distance",
    # Technical
    "Ret_1", "Ret_3", "Ret_5", "Ret_10", "Ret_20",
    "MA_dist_5", "MA_dist_10", "MA_dist_20", "MA_dist_50",
    "MACD_Hist", "RSI", "StochRSI",
    "BB_Position", "BB_Width",
    "ATR_pct", "Vol_5_21_Ratio", "Vol_Ratio", "Trend_Strength",
    # Breadth
    "ADR", "Breadth_Net", "Top_Gainer_Ret", "Top_Loser_Ret",
    "Gainer_Loser_Spread", "ADR_MA3", "Breadth_Net_MA3",
]


def build_feature_matrix(filepath: Path | None = None,
                         start_date: str = "2015-01-01") -> pd.DataFrame:
    """
    End-to-end feature building pipeline.

    Returns a clean DataFrame with FEATURE_COLS + Target column,
    ready for model training.
    """
    print("📂 Loading data …")
    raw = load_nifty50_all(filepath)
    raw = raw[raw["Date"] >= pd.Timestamp(start_date)]

    print(f"   {len(raw):,} rows | {raw['Symbol'].nunique()} symbols | "
          f"{raw['Date'].min().date()} → {raw['Date'].max().date()}")

    print("📈 Building synthetic NIFTY index …")
    idx = build_nifty_index(raw)

    print("🔧 Engineering option-chain features …")
    idx = simulate_option_chain_features(idx)

    print("📊 Adding technical indicators …")
    idx = add_technical_features(idx)

    print("🗂️  Adding breadth features …")
    idx = add_breadth_features(idx, raw)

    print("🏷️  Adding target label …")
    idx = add_target(idx)

    # Drop rows where features or target are NaN
    available = [c for c in FEATURE_COLS if c in idx.columns]
    idx = idx.dropna(subset=available + ["Target"])
    idx = idx.sort_values("Date").reset_index(drop=True)

    print(f"✅ Feature matrix ready: {len(idx):,} rows × {len(available)} features")
    class_dist = idx["Target"].value_counts().sort_index()
    labels = {0: "Down", 1: "Sideways", 2: "Up"}
    for k, v in class_dist.items():
        print(f"   Class {k} ({labels[k]}): {v:,}  ({v/len(idx)*100:.1f}%)")

    return idx[["Date", "Spot", "Fwd_Return"] + available + ["Target"]]


if __name__ == "__main__":
    df = build_feature_matrix()
    out = Path(__file__).parent / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved → {out}")

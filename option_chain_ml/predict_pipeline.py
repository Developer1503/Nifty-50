"""
predict_pipeline.py
====================
Real-time / live prediction pipeline for NIFTY-50 market direction.

Modes:
  1. BACKTEST MODE  — feed historical rows one at a time and get predictions.
  2. LIVE MODE      — fetch current data from NSE India unofficial APIs
                      or any data provider and make a real-time prediction.

Usage:
    # Backtest last 30 days:
    python option_chain_ml/predict_pipeline.py --mode backtest --days 30

    # One-shot live prediction (requires internet):
    python option_chain_ml/predict_pipeline.py --mode live

The live pipeline:
    1. Downloads the latest NSE option chain from the public NSE endpoint.
    2. Extracts real PCR, Max Pain, IV from the JSON response.
    3. Runs the same feature vector through the saved model.
    4. Prints the prediction with confidence.
"""

import argparse
import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests
import joblib

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
ML_DIR  = Path(__file__).resolve().parent
MODELS  = ML_DIR / "models"
sys.path.insert(0, str(ML_DIR))
from feature_engineering import (
    build_feature_matrix, FEATURE_COLS, load_nifty50_all,
    build_nifty_index, simulate_option_chain_features,
    add_technical_features, add_breadth_features,
)

LABEL_MAP   = {0: "📉 DOWN", 1: "↔️  SIDEWAYS", 2: "📈 UP"}
LABEL_COLOR = {0: "\033[91m", 1: "\033[93m", 2: "\033[92m"}  # red, yellow, green
RESET       = "\033[0m"


# ═══════════════════════════════════════════════════════════════════════════
# LIVE NSE OPTION-CHAIN FETCHER
# ═══════════════════════════════════════════════════════════════════════════

NSE_BASE = "https://www.nseindia.com"
NSE_OC   = "/api/option-chain-indices?symbol=NIFTY"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}


def fetch_nse_option_chain(timeout: int = 15) -> dict | None:
    """
    Fetch NIFTY option chain from NSE India public API.

    NSE requires a session cookie obtained from the home page first.
    Returns parsed JSON or None on failure.
    """
    session = requests.Session()
    try:
        # Step 1: get cookies
        session.get(NSE_BASE, headers=HEADERS, timeout=timeout)
        time.sleep(1)
        # Step 2: fetch option chain
        resp = session.get(NSE_BASE + NSE_OC, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"⚠  NSE fetch failed: {e}")
        return None


def parse_option_chain(data: dict) -> dict:
    """
    Parse the NSE JSON response and compute:
        - spot_price
        - put_call_ratio (PCR by OI)
        - max_pain_strike
        - atm_iv  (ATM implied volatility average CE+PE)
        - total_ce_oi, total_pe_oi
        - top 5 CE and PE strikes by OI
    """
    if data is None:
        return {}

    filtered = data.get("filtered", {})
    records  = filtered.get("data", [])

    spot = data.get("records", {}).get("underlyingValue", np.nan)

    total_ce_oi = 0
    total_pe_oi = 0
    atm_iv_vals = []
    oi_by_strike: dict[float, dict] = {}

    atm_strike = round(spot / 50) * 50 if not np.isnan(spot) else None

    for rec in records:
        strike = rec.get("strikePrice", 0)

        ce = rec.get("CE", {})
        pe = rec.get("PE", {})

        ce_oi = ce.get("openInterest",       0) or 0
        pe_oi = pe.get("openInterest",       0) or 0
        ce_iv = ce.get("impliedVolatility",  0) or 0
        pe_iv = pe.get("impliedVolatility",  0) or 0

        total_ce_oi += ce_oi
        total_pe_oi += pe_oi

        if atm_strike and abs(strike - atm_strike) <= 100:
            if ce_iv > 0:
                atm_iv_vals.append(ce_iv)
            if pe_iv > 0:
                atm_iv_vals.append(pe_iv)

        oi_by_strike[strike] = {"ce_oi": ce_oi, "pe_oi": pe_oi}

    pcr = total_pe_oi / (total_ce_oi + 1e-9)
    atm_iv = np.mean(atm_iv_vals) if atm_iv_vals else np.nan

    # Max Pain: strike where total payout to options buyers is minimised
    max_pain = compute_max_pain(oi_by_strike)

    # Top OI strikes
    top_ce = sorted(oi_by_strike, key=lambda k: oi_by_strike[k]["ce_oi"], reverse=True)[:5]
    top_pe = sorted(oi_by_strike, key=lambda k: oi_by_strike[k]["pe_oi"], reverse=True)[:5]

    return {
        "spot":        spot,
        "pcr":         pcr,
        "atm_iv":      atm_iv,
        "max_pain":    max_pain,
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
        "top_ce_oi_strikes": top_ce,
        "top_pe_oi_strikes": top_pe,
    }


def compute_max_pain(oi_by_strike: dict) -> float:
    """
    Max Pain = strike where total in-the-money OI value is minimised.
    (Sellers want price to land here — minimum payout to buyers.)
    """
    strikes = sorted(oi_by_strike.keys())
    if not strikes:
        return 0.0

    payouts = {}
    for s in strikes:
        total_payout = 0.0
        for k, v in oi_by_strike.items():
            # CE holders profit if expiry > k  → payout = (s - k) * ce_oi  for k < s
            if s > k:
                total_payout += (s - k) * v["ce_oi"]
            # PE holders profit if expiry < k  → payout = (k - s) * pe_oi  for k > s
            if s < k:
                total_payout += (k - s) * v["pe_oi"]
        payouts[s] = total_payout

    return min(payouts, key=payouts.get)


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST MODE
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(best_model_name: str = "random_forest", last_n_days: int = 30):
    """
    Re-run predictions on the last N days of historical data.
    Loads the saved model and feature matrix.
    """
    print("\n" + "═" * 60)
    print("  BACKTEST MODE")
    print("═" * 60)

    # Load model
    model_path = MODELS / f"{best_model_name}.pkl"
    if not model_path.exists():
        print(f"❌  Model not found: {model_path}")
        print("    Run train_models.py first.")
        return

    model = joblib.load(model_path)
    print(f"  Loaded model : {model_path.name}")

    # Build full feature matrix
    df = build_feature_matrix(start_date="2015-01-01")
    available = [c for c in FEATURE_COLS if c in df.columns]

    recent = df.tail(last_n_days).copy()
    X_recent = recent[available]
    y_actual = recent["Target"].values

    preds  = model.predict(X_recent)
    probas = model.predict_proba(X_recent)

    print(f"\n  Last {last_n_days} trading-day predictions:\n")
    print(f"  {'Date':<12}  {'Spot':>8}  {'Actual':^9}  {'Predicted':^9}  "
          f"{'Down%':>6}  {'Sidew%':>7}  {'Up%':>5}")
    print(f"  {'─'*70}")

    correct = 0
    for i, (_, row) in enumerate(recent.iterrows()):
        actual_lbl = LABEL_MAP[y_actual[i]].split()[1]
        pred_lbl   = LABEL_MAP[preds[i]].split()[1]
        match = "✓" if preds[i] == y_actual[i] else "✗"
        if preds[i] == y_actual[i]:
            correct += 1
        p = probas[i]
        print(f"  {str(row['Date'].date()):<12}  "
              f"{row['Spot']:>8.0f}  "
              f"{actual_lbl:^9}  "
              f"{pred_lbl:^9}  "
              f"{p[0]*100:>5.1f}%  {p[1]*100:>5.1f}%  {p[2]*100:>5.1f}%  {match}")

    acc = correct / len(recent)
    print(f"\n  Accuracy on last {last_n_days} days: {acc:.2%}  ({correct}/{len(recent)})")


# ═══════════════════════════════════════════════════════════════════════════
# LIVE PREDICTION MODE
# ═══════════════════════════════════════════════════════════════════════════

def run_live_prediction(best_model_name: str = "random_forest"):
    """
    Fetch live NSE option chain + latest OHLCV, build features, predict.
    """
    print("\n" + "═" * 60)
    print("  LIVE PREDICTION MODE")
    print("═" * 60)

    model_path = MODELS / f"{best_model_name}.pkl"
    if not model_path.exists():
        print(f"❌  Model not found. Run train_models.py first.")
        return

    model = joblib.load(model_path)
    print(f"  Loaded model   : {model_path.name}")
    print(f"  Timestamp      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Fetch live option chain ───────────────────────────────────────────
    print("\n  🌐 Fetching NSE option chain …")
    oc_data  = fetch_nse_option_chain()
    oc_stats = parse_option_chain(oc_data)

    if not oc_stats:
        print("  ⚠  Could not fetch live data. Falling back to last historical row.")
        df = build_feature_matrix(start_date="2020-01-01")
        available = [c for c in FEATURE_COLS if c in df.columns]
        row = df.tail(1)[available]
        spot = df.tail(1)["Spot"].iloc[0]
        oc_stats = {"spot": spot, "pcr": np.nan, "atm_iv": np.nan,
                    "max_pain": np.nan}
    else:
        # ── Build historical feature context ─────────────────────────────
        print("  📊 Building feature context from historical data …")
        df = build_feature_matrix(start_date="2020-01-01")
        available = [c for c in FEATURE_COLS if c in df.columns]

        # Override option-chain features with live values
        row = df.tail(1)[available].copy()
        if not np.isnan(oc_stats.get("pcr", np.nan)):
            if "PCR" in row.columns:
                row["PCR"] = oc_stats["pcr"]
        if not np.isnan(oc_stats.get("atm_iv", np.nan)):
            if "IV" in row.columns:
                row["IV"] = oc_stats["atm_iv"]
        if not np.isnan(oc_stats.get("max_pain", np.nan)) and \
                not np.isnan(oc_stats.get("spot", np.nan)):
            if "Max_Pain_Distance" in row.columns:
                row["Max_Pain_Distance"] = (
                    (oc_stats["spot"] - oc_stats["max_pain"]) /
                    oc_stats["spot"]
                )
        spot = oc_stats.get("spot", df["Spot"].iloc[-1])

    pred   = model.predict(row)[0]
    proba  = model.predict_proba(row)[0]

    # ── Display results ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  📋  LIVE OPTION CHAIN SNAPSHOT")
    print("─" * 60)
    if oc_stats:
        print(f"  NIFTY Spot   : {oc_stats.get('spot', 'N/A')}")
        print(f"  PCR (OI)     : {oc_stats.get('pcr', 'N/A'):.3f}" if oc_stats.get("pcr") else "  PCR          : N/A")
        print(f"  ATM IV       : {oc_stats.get('atm_iv', 'N/A'):.2f}%" if oc_stats.get("atm_iv") else "  ATM IV       : N/A")
        print(f"  Max Pain     : {oc_stats.get('max_pain', 'N/A')}")
        print(f"  Total CE OI  : {oc_stats.get('total_ce_oi', 0):,.0f}")
        print(f"  Total PE OI  : {oc_stats.get('total_pe_oi', 0):,.0f}")
        if oc_stats.get("top_ce_oi_strikes"):
            print(f"  Top CE OI strikes : {oc_stats['top_ce_oi_strikes']}")
        if oc_stats.get("top_pe_oi_strikes"):
            print(f"  Top PE OI strikes : {oc_stats['top_pe_oi_strikes']}")

    print("\n" + "─" * 60)
    print("  🤖  MODEL PREDICTION")
    print("─" * 60)

    color = LABEL_COLOR.get(pred, "")
    print(f"\n  Direction : {color}{LABEL_MAP[pred]}{RESET}")
    print(f"\n  Confidence breakdown:")
    for cls, prob in enumerate(proba):
        bar = "█" * int(prob * 30)
        print(f"    {LABEL_MAP[cls]:18s}  {prob*100:5.1f}%  {bar}")

    # Interpretation guide
    print("\n  📖 Interpretation:")
    if oc_stats.get("pcr"):
        pcr = oc_stats["pcr"]
        if pcr > 1.3:
            print(f"    PCR = {pcr:.2f} → Elevated put buying → bearish sentiment / potential reversal up")
        elif pcr < 0.7:
            print(f"    PCR = {pcr:.2f} → Elevated call buying → bullish sentiment / potential reversal down")
        else:
            print(f"    PCR = {pcr:.2f} → Neutral")

    print("\n" + "─" * 60)
    print("  ⚠  DISCLAIMER: For educational purposes only. Not financial advice.")
    print("─" * 60)

    return {"prediction": int(pred), "proba": proba.tolist(), "spot": spot}


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NIFTY-50 ML Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["backtest", "live"], default="backtest",
        help="backtest = historical validation, live = real-time NSE data"
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Number of past days to show in backtest mode"
    )
    parser.add_argument(
        "--model", default="random_forest",
        choices=["logistic_regression", "random_forest", "xgboost"],
        help="Which saved model to use for prediction"
    )
    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(best_model_name=args.model, last_n_days=args.days)
    else:
        run_live_prediction(best_model_name=args.model)


if __name__ == "__main__":
    main()

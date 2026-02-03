# ============================================================
# Dash Portfolio KPI Dashboard (Google Colab - FIXED)
# ============================================================

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc

# ------------------------------------------------------------
# 1. Portfolio Returns (replace with real strategy output later)
# ------------------------------------------------------------
np.random.seed(42)

portfolio_returns = np.random.normal(
    loc=-0.00005,   # aligns with Sharpe ≈ -0.03
    scale=0.0049,   # ~7.8% annual volatility
    size=800
)

dates = pd.date_range(start="2018-01-01", periods=len(portfolio_returns))

df = pd.DataFrame({
    "Date": dates,
    "Daily_Return": portfolio_returns
})

# ------------------------------------------------------------
# 2. KPI Calculations
# ------------------------------------------------------------
equity_curve = (1 + df["Daily_Return"]).cumprod()

total_return = equity_curve.iloc[-1] - 1
annual_volatility = df["Daily_Return"].std() * np.sqrt(252)
sharpe_ratio = (
    df["Daily_Return"].mean()
    / df["Daily_Return"].std()
    * np.sqrt(252)
)

drawdown = equity_curve / equity_curve.cummax() - 1

# ------------------------------------------------------------
# 3. Dash App
# ------------------------------------------------------------
app = Dash(__name__)

app.layout = html.Div(
    style={"fontFamily": "Arial", "padding": "20px"},
    children=[

        html.H1("📊 ML Multi-Stock Portfolio Dashboard"),

        # ---------------- KPIs ----------------
        html.Div(
            style={"display": "flex", "gap": "40px"},
            children=[
                html.Div([
                    html.H4("📈 Total Return"),
                    html.H2(f"{total_return*100:.2f}%")
                ]),
                html.Div([
                    html.H4("⚡ Annual Volatility"),
                    html.H2(f"{annual_volatility*100:.2f}%")
                ]),
                html.Div([
                    html.H4("📊 Sharpe Ratio"),
                    html.H2(f"{sharpe_ratio:.2f}")
                ]),
            ]
        ),

        html.Hr(),

        # ---------------- Equity Curve ----------------
        html.H3("📈 Portfolio Equity Curve"),
        dcc.Graph(
            figure=px.line(
                x=df["Date"],
                y=equity_curve,
                labels={"x": "Date", "y": "Portfolio Value"}
            )
        ),

        # ---------------- Drawdown ----------------
        html.H3("📉 Drawdown"),
        dcc.Graph(
            figure=px.area(
                x=df["Date"],
                y=drawdown,
                labels={"x": "Date", "y": "Drawdown"}
            )
        ),

        # ---------------- Return Distribution ----------------
        html.H3("📊 Daily Returns Distribution"),
        dcc.Graph(
            figure=px.histogram(
                df,
                x="Daily_Return",
                nbins=50
            )
        ),
    ]
)

# ------------------------------------------------------------
# 4. Run Dash INLINE in Colab (CORRECT METHOD)
# ------------------------------------------------------------
app.run(jupyter_mode="inline")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# 1. Hardcoded Tickers (3 per GICS Sector)
# This ensures total diversification across the entire economy.
sectors = {
    "Technology": ["AAPL", "MSFT", "NVDA"],
    "Financials": ["JPM", "V", "MA"],
    "Healthcare": ["LLY", "UNH", "JNJ"],
    "Cons. Disc": ["AMZN", "TSLA", "HD"],
    "Communication": ["GOOGL", "META", "NFLX"],
    "Industrials": ["CAT", "GE", "UNP"],
    "Cons. Staples": ["PG", "KO", "PEP"],
    "Energy": ["XOM", "CVX", "COP"],
    "Utilities": ["NEE", "SO", "DUK"],
    "Real Estate": ["PLD", "AMT", "EQIX"],
    "Materials": ["LIN", "SHW", "APD"]
}

# Flatten the dictionary into a list for downloading
tickers = [ticker for sublist in sectors.values() for ticker in sublist]

# 2. Downloading Data
print(f"Step 1: Downloading data for {len(tickers)} diversified assets...")
start_date = '2021-01-01'
end_date = '2026-01-01'
raw_data = pd.DataFrame()

for t in tickers:
    try:
        # auto_adjust=True handles dividends and stock splits
        df = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if not df.empty:
            # iloc[:, 0] ensures we grab the price regardless of column naming
            raw_data[t] = df.iloc[:, 0]
    except Exception as e:
        print(f"Could not download {t}: {e}")
        continue

# Clean the data (remove any stocks that failed to download)
data = raw_data.dropna(axis=1, how='all').dropna()
final_tickers = data.columns.tolist()
print(f"Successfully loaded {len(final_tickers)} tickers.")

# 3. Risk/Return Calculations
returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# 4. Optimization Functions
def get_metrics(w):
    p_ret = np.sum(mean_returns * w)
    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    return p_ret, p_vol

def min_obj_sharpe(w):
    p_ret, p_vol = get_metrics(w)
    # Using 4% as a proxy for the risk-free

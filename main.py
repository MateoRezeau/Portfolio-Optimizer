import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# 1. Setup
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM']
start_date = '2021-01-01'
end_date = '2026-01-01'

# 2. Download - THE NEW WAY
print("Downloading data...")
# We use auto_adjust=True so 'Adj Close' is merged into 'Close'
raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

# If yfinance gives a MultiIndex, we extract the 'Close' level
if isinstance(raw.columns, pd.MultiIndex):
    data = raw['Close']
else:
    data = raw

data = data[tickers].dropna()
print(f"Success! Data for {len(data.columns)} stocks loaded.")

# 3. Simple MVO Logic
returns = data.pct_change().dropna()
mean_ret = returns.mean() * 252
cov_mat = returns.cov() * 252

def get_perf(w):
    p_ret = np.sum(mean_ret * w)
    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    return p_ret, p_vol

def min_func_sharpe(w):
    p_ret, p_vol = get_perf(w)
    return -(p_ret - 0.04) / p_vol

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for _ in range(len(tickers)))
res = minimize(min_func_sharpe, [1./6]*6, method='SLSQP', bounds=bnds, constraints=cons)

print("\n--- OPTIMAL WEIGHTS ---")
for i, t in enumerate(tickers):
    print(f"{t}: {res.x[i]:.2%}")

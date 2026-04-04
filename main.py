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

# Flatten the dictionary into a simple list
tickers = [ticker for sublist in sectors.values() for ticker in sublist]

# 2. Downloading Data
print(f"Step 1: Downloading data for {len(tickers)} diversified assets...")
start_date = '2021-01-01'
end_date = '2026-01-01'
raw_data = pd.DataFrame()

for t in tickers:
    try:
        # Using the .iloc[:, 0] trick to avoid naming errors
        df = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if not df.empty:
            raw_data[t] = df.iloc[:, 0]
    except:
        continue

data = raw_data.dropna(axis=1, how='all').dropna()
final_tickers = data.columns.tolist()
print(f"Successfully loaded {len(final_tickers)} tickers.")

# 3. Calculations
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
    return -(p_ret - 0.04) / p_vol # 4% Risk-Free Rate

# 5. Optimizer Configuration
# Constraint: Weights must sum to 100%
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds: No short selling (0) and Max 10% per stock for safety
bounds = tuple((0, 0.10) for _ in range(len(final_tickers)))

initial_guess = [1. / len(final_tickers)] * len(final_tickers)

# 6. Run Optimizer
print("\nStep 2: Solving for Optimal Weights (Max 10% per asset)...")
res = minimize(min_obj_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

if res.success:
    print("\n" + "="*45)
    print("      DIVERSIFIED SECTOR OPTIMAL PORTFOLIO")
    print("="*45)
    
    # Create results table
    results = pd.DataFrame({'Ticker': final_tickers, 'Weight': res.x})
    
    # Find sector for each ticker for the final report
    ticker_to_sector = {t: s for s, t_list in sectors.items() for t in t_list}
    results['Sector'] = results['Ticker'].map(ticker_to_sector)
    
    # Filter for weights > 0.1% and sort
    results = results[results['Weight'] > 0.00

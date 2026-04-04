import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# 1. Hardcoded Tickers (3 per GICS Sector)
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

# Flatten the dictionary into a list
tickers = [ticker for sublist in sectors.values() for ticker in sublist]

# 2. Downloading Data
print(f"Step 1: Downloading data for {len(tickers)} diversified assets...")
start_date = '2021-01-01'
end_date = '2026-01-01'
raw_data = pd.DataFrame()

for t in tickers:
    try:
        df = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if not df.empty:
            raw_data[t] = df.iloc[:, 0]
    except Exception as e:
        print(f"Skipping {t}: {e}")
        continue

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
    return -(p_ret - 0.04) / p_vol

# 5. Constraints and Bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 0.10) for _ in range(len(final_tickers)))
initial_guess = [1. / len(final_tickers)] * len(final_tickers)

# 6. Run Optimizer
print("\nStep 2: Solving for Optimal Weights (Max 10% per asset)...")
res = minimize(min_obj_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

if res.success:
    print("\n" + "="*50)
    print("      DIVERSIFIED SECTOR OPTIMAL PORTFOLIO")
    print("="*50)
    
    # Create results table
    opt_weights = res.x
    results_df = pd.DataFrame({'Ticker': final_tickers, 'Weight': opt_weights})
    
    # Map tickers back to their sectors
    ticker_to_sector = {t: s for s, t_list in sectors.items() for t in t_list}
    results_df['Sector'] = results_df['Ticker'].map(ticker_to_sector)
    
    # --- FIXED SECTION ---
    # Filter for weights > 0.1% and sort
    results_df = results_df[results_df['Weight'] > 0.001].sort_values(by='Weight', ascending=False)
    
    print(f"{'TICKER':<8} | {'SECTOR':<15} | {'WEIGHT'}")
    print("-" * 50)
    for _, row in results_df.iterrows():
        print(f"{row['Ticker']:<8} | {row['Sector']:<15} | {row['Weight']:.2%}")
    
    p_ret, p_vol = get_metrics(opt_weights)
    print("-" * 50)
    print(f"Expected Annual Return: {p_ret:.2%}")
    print(f"Annual Volatility:     {p_vol:.2%}")
    print(f"Sharpe Ratio:          {(-res.fun):.2f}")
    print("="*50)
else:
    print(f"Optimizer failed: {res.message}")

print("\n[Script Execution Finished Successfully]")

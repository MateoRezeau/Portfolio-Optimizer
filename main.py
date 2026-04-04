import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import os

# 1. Configuration & Sector Selection
print("Step 1: Fetching S&P 500 Sector Structure...")
try:
    # Scraping the current S&P 500 list from Wikipedia
    sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sp500_table['Symbol'] = sp500_table['Symbol'].str.replace('.', '-', regex=False)
    
    # Selecting the Top 3 companies from each of the 11 GICS Sectors
    selected_df = sp500_table.groupby('GICS Sector').head(3)
    tickers = selected_df['Symbol'].tolist()
    print(f"Targeting {len(tickers)} assets across all market sectors.")
except Exception as e:
    print(f"Wikipedia Scraping failed: {e}")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM', 'PG'] # Fallback

# 2. Downloading Data
print("\nStep 2: Downloading Historical Price Data...")
start_date = '2021-01-01'
end_date = '2026-01-01'
raw_data = pd.DataFrame()

for t in tickers:
    try:
        # Download 1-by-1 to avoid MultiIndex KeyError
        df = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if not df.empty:
            # iloc[:, 0] ensures we grab price regardless of the column name
            raw_data[t] = df.iloc[:, 0]
    except:
        continue

# Clean and verify
data = raw_data.dropna(axis=1, how='all').dropna()
final_tickers = data.columns.tolist()
print(f"Successfully loaded {len(final_tickers)} tickers.")

# 3. Risk/Return Calculations
returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# 4. Optimization Functions
def get_portfolio_metrics(weights):
    p_ret = np.sum(mean_returns * weights)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return p_ret, p_vol

def min_obj_sharpe(weights):
    p_ret, p_vol = get_portfolio_metrics(weights)
    return -(p_ret - 0.04) / p_vol  # 4% Risk-Free Rate

# 5. The Optimizer with Professional Constraints
# Constraint 1: Weights must sum to 1.0 (100%)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Constraint 2: No short selling (0%) and Max 10% per stock for diversification
# Professional portfolios rarely allow a single stock to dominate the risk.
bounds = tuple((0, 0.10) for _ in range(len(final_tickers)))

initial_guess = [1. / len(final_tickers)] * len(final_tickers)

print("\nStep 3: Solving for Optimal Weights...")
optimized = minimize(
    min_obj_sharpe, 
    initial_guess, 
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

# 6. Output Results
if optimized.success:
    opt_weights = optimized.x
    p_ret, p_vol = get_portfolio_metrics(opt_weights)
    
    print("\n" + "="*45)
    print("      STRATIFIED S&P 500 OPTIMAL PORTFOLIO")
    print("="*45)
    
    # Only print tickers with a weight > 0.1% to keep it clean
    results_df = pd.DataFrame({'Ticker': final_tickers, 'Weight': opt_weights})
    results_df = results_df[results_df['Weight'] > 0.001].sort_values(by='Weight', ascending=False)
    
    for index, row in results_df.iterrows():
        print(f"{row['Ticker']:8} | {row['Weight']:.2%}")
        
    print("-" * 45)
    print(f"Expected Annual Return: {p_ret:.2%}")
    print(f"Annual Volatility (Risk): {p_vol:.2%}")
    print(f"Sharpe Ratio: {(-optimized.fun):.2f}")
    print("="*45)
else:
    print("Optimizer failed to find a solution.")

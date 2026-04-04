import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# 1. Configuration
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM']
start_date = '2021-01-01'
end_date = '2026-01-01'
risk_free_rate = 0.04 

# 2. Download Data (The "Blind" Failsafe)
data = pd.DataFrame()
print("Fetching market data...")
for ticker in tickers:
    # We download one by one to avoid the MultiIndex mess
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if not df.empty:
        # EXPLANATION: .iloc[:, 0] means "Take the first column of data" 
        # regardless of whether its name is 'Close', 'Adj Close', or 'Price'.
        data[ticker] = df.iloc[:, 0] 
# Drop any days where data is missing
data = data.dropna()
if data.empty:
    print("ERROR: Dataframe is empty. Check your internet or tickers.")
else:
    print(f"SUCCESS: Data loaded for {list(data.columns)}")
    print(data.head(2))

# 3. Basic Calculations
log_returns = np.log(data / data.shift(1)).dropna()
cov_matrix = log_returns.cov() * 252
mean_returns = log_returns.mean() * 252

# 4. Optimization Functions
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std

# 5. The Optimizer
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in range(len(tickers)))
initial_guess = len(tickers) * [1. / len(tickers)]

optimized_results = minimize(
    negative_sharpe_ratio, 
    initial_guess, 
    args=(mean_returns, cov_matrix, risk_free_rate),
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

# 6. Results Output
optimal_weights = optimized_results.x
perf = portfolio_performance(optimal_weights, mean_returns, cov_matrix)

print("\n" + "="*30)
print("OPTIMIZED PORTFOLIO WEIGHTS")
print("="*30)
for i, ticker in enumerate(tickers):
    print(f"{ticker:7}: {optimal_weights[i]:.2%}")
print("-" * 30)
print(f"Expected Return: {perf[0]:.2%}")
print(f"Annual Risk:    {perf[1]:.2%}")
print(f"Sharpe Ratio:   {(-optimized_results.fun):.2f}")

# 7. Visualization
if not os.path.exists('results'):
    os.makedirs('results')

plt.figure(figsize=(10, 6))
plt.bar(tickers, optimal_weights, color='skyblue')
plt.title("Optimal Asset Allocation")
plt.ylabel("Weighting")
plt.savefig("results/allocation.png")
print("\nSuccess! Chart saved to results/allocation.png")

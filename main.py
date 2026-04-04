import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Configuration
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM']
start_date = '2021-01-01'
end_date = '2026-01-01'
risk_free_rate = 0.04  # Assuming 4% risk-free rate in 2026

# 2. Download Data
# This downloads all data and then "cuts" through the levels to get only what we need.
raw_data = yf.download(tickers, start=start_date, end=end_date)
# This single line handles the MultiIndex error:
# It looks for 'Adj Close' across all tickers, regardless of the table structure.
if 'Adj Close' in raw_data.columns.get_level_values(0):
    data = raw_data.xs('Adj Close', axis=1, level=0)
else:
    data = raw_data.xs('Close', axis=1, level=0)
# Clean up: Ensure we only have our tickers and no missing values
data = data[tickers].dropna()
print("-" * 30)
print(f"SYSTEM CHECK: Data successfully extracted for {len(data.columns)} assets.")
print(data.head(3)) # This proves the table is now "flat" and readable

# 3. Basic Calculations
log_returns = np.log(data / data.shift(1)).dropna()
cov_matrix = log_returns.cov() * 252 # Annualized covariance
mean_returns = log_returns.mean() * 252 # Annualized mean returns

# 4. Functions for Optimization
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std

# 5. The Optimizer
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Weights sum to 1
bounds = tuple((0, 1) for asset in range(len(tickers))) # No short selling
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

print("-" * 30)
print("OPTIMIZED PORTFOLIO WEIGHTS")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: {optimal_weights[i]:.2%}")
print("-" * 30)
print(f"Expected Annual Return: {perf[0]:.2%}")
print(f"Annual Volatility: {perf[1]:.2%}")
print(f"Sharpe Ratio: {(-optimized_results.fun):.2f}")

# 7. Visualization (Add this to the bottom of main.py)
plt.figure(figsize=(10, 6))
# (Logic to plot the frontier goes here...)
plt.title("Efficient Frontier & Optimal Portfolio")
plt.xlabel("Volatility (Risk)")
plt.ylabel("Expected Return")
plt.savefig("results/efficient_frontier.png") # This saves the image

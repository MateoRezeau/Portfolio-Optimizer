import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import urllib.request

# 1. Configuration & Sector Selection (With Browser Headers)
print("Step 1: Fetching S&P 500 Sector Structure...")
try:
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    # Disguise the request as a Chrome browser to avoid 403 Forbidden
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        sp500_table = pd.read_html(response)[0]
    
    sp500_table['Symbol'] = sp500_table['Symbol'].str.replace('.', '-', regex=False)
    selected_df = sp500_table.groupby('GICS Sector').head(3)
    tickers = selected_df['Symbol'].tolist()
    print(f"Targeting {len(tickers)} assets from Wikipedia list.")
except Exception as e:
    print(f"Scraping failed again: {e}. Using large fallback list...")
    # Expanded fallback ensures we can reach 100% even with a 10% cap
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B', 'JPM', 'V', 'UNH', 'MA', 'PG', 'HD', 'LLY']

# 2. Downloading Data
print("\nStep 2: Downloading Historical Price Data...")
start_date = '2021-01-01'
end_date = '2026-01-01'
raw_data = pd.DataFrame()

for t in tickers:
    try:
        df = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if not df.empty:
            raw_data[t] = df.iloc[:, 0]
    except:
        continue

data = raw_data.dropna(axis=1, how='all').dropna()
final_tickers = data.columns.tolist()
num_assets = len(final_tickers)
print(f"Successfully loaded {num_assets} tickers.")

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
    return -(p_ret - 0.04) / p_vol

# 5. The Optimizer with Logic Safety
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# FEASIBILITY CHECK: 
# If we have 15+ stocks, we cap at 10%. 
# If we have 5-14 stocks, we cap at 20%.
# If we have fewer, we allow 100%.
if num_assets >= 15:
    max_w = 0.10
elif num_assets >= 5:
    max_w = 0.20
else:
    max_w = 1.0

bounds = tuple((0, max_w) for _ in range(num_assets))
print(f"Applying {max_w*100}% max weight limit per asset.")
initial_guess = [1. / num_assets] * num_assets

# 6. Run
res = minimize(min_obj_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

if res.success:
    print("\n" + "="*45)
    print("      S&P 500 SECTOR OPTIMAL PORTFOLIO")
    print("="*45)
    results = pd.DataFrame({'Ticker': final_tickers, 'Weight': res.x})
    results = results[results['Weight'] > 0.001].sort_values(by='Weight', ascending=False)
    
    for _, row in results.iterrows():
        print(f"{row['Ticker']:8} | {row['Weight']:.2%}")
    
    p_ret, p_vol = get_metrics(res.x)
    print("-" * 45)
    print(f"Expected Annual Return: {p_ret:.2%}")
    print(f"Annual Volatility:     {p_vol:.2%}")
    print(f"Sharpe Ratio:          {(-res.fun):.2f}")
    print("="*45)
else:
    print(f"Optimizer failed: {res.message}")

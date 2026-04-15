# S&P 500 Sector-Stratified Portfolio Optimizer

## Project Overview
This project implements a Mean-Variance Optimization (MVO) model to build a diversified "Proxy Portfolio" of the S&P 500. Instead of a random selection, the algorithm targets the top 3 market-cap leaders from each of the 12 GICS sectors, ensuring the portfolio remains balanced across the entire US economy.

## Technical Features
- **Data Engine:** Integrated with `yfinance` to pull 5 years of historical adjusted closing prices.
- **Optimization Strategy:** Maximizes the Sharpe Ratio using the `SLSQP` algorithm.
- **Institutional Constraints:** Enforces a 10.00% maximum weight per asset to prevent single-stock concentration risk and improve diversification.
- **Error Handling:** Robust processing to handle API timeouts and missing ticker data.

## Performance Results (2021 - 2026)
As of 15/04/2026, the model identified the following optimal allocation to maximize returns while maintaining a 15% volatility target:

| TICKER | SECTOR | WEIGHT |
| :--- | :--- | :--- |
| **NVDA** | Technology | 10.00% |
| **XOM** | Energy | 10.00% |
| **GOOGL** | Communication | 10.00% |
| **GE** | Industrials | 10.00% |
| **LLY** | Healthcare | 10.00% |
| **SO** | Utilities | 10.00% |
| **DUK** | Utilities | 10.00% |
| **JNJ** | Healthcare | 10.00% |
| **CAT** | Industrials | 7.45% |
| **KO** | Cons. Staples | 6.20% |
| **JPM** | Financials | 5.25% |
| **COP** | Energy | 1.10% |

### Key Metrics:
- **Expected Annual Return:** 28.93%
- **Annual Volatility (Risk):** 15.00%
- **Sharpe Ratio:** 1.66

## Installation & Usage
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git`
2. Install dependencies: `pip install numpy pandas yfinance scipy`
3. Run the optimizer: `python main.py`

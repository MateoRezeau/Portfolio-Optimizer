# S&P 500 Sector-Stratified Portfolio Optimizer

An automated tool that uses **Mean-Variance Optimization (MVO)** to build a diversified portfolio. It selects the top 3 companies from each of the 11 GICS sectors and calculates the optimal weights to maximize the **Sharpe Ratio**.

## Key Features
- **Data Sourcing:** Live market data via `yfinance` API.
- **Risk Management:** Implements a **10% maximum position limit** to ensure institutional-grade diversification.
- **Optimization:** Uses the `scipy.optimize` (SLSQP) engine to find the highest return-to-risk ratio.

## Latest Results (2021 - 2026)
| Ticker | Sector | Weight |
| :--- | :--- | :--- |
| **NVDA** | Technology | 10.00% |
| **LLY** | Healthcare | 10.00% |
| **JNJ** | Healthcare | 10.00% |
| **GE** | Industrials | 10.00% |
| **DUK** | Utilities | 10.00% |
| **GOOGL** | Communication | 10.00% |
| **SO** | Utilities | 10.00% |
| **XOM** | Energy | 10.00% |
| **CAT** | Industrials | 7.45% |
| **KO** | Cons. Staples | 6.20% |

**Performance Metrics:**
- **Expected Annual Return:** 28.93%
- **Annual Volatility:** 15.00%
- **Sharpe Ratio:** 1.66
